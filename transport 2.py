import os
import json
import time
import logging
from typing import Dict, Any, List, Optional
from requests_aws4auth import AWS4Auth
from opensearchpy import OpenSearch, RequestsHttpConnection
import boto3
from opensearchpy import OpenSearch

from strands import Agent
from strands.models import BedrockModel
from strands.tools import tool
from strands.tools.executors import ConcurrentToolExecutor
from strands.hooks import HookProvider, HookRegistry, BeforeInvocationEvent, AfterInvocationEvent
from opensearchpy import OpenSearch

import dotenv
dotenv.load_dotenv()

# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------
logging.basicConfig(
    format="%(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("multiagent-system")
logger.setLevel(logging.INFO)

session = boto3.Session()
credentials = session.get_credentials()
AWS_REGION = "ap-south-1"  # Glue / Athena / OpenSearch region

awsauth = AWS4Auth(
    credentials.access_key,
    credentials.secret_key,
    AWS_REGION,
    "es",
    session_token=credentials.token
)


# ---------------------------------------------------------
# AWS Clients / Config
# ---------------------------------------------------------
session = boto3.Session(region_name=AWS_REGION)
glue = session.client("glue")
athena = session.client("athena")

OPENSEARCH_ENDPOINT = "vpc-volty-os-lambda-prod-6ygvvfidbfj3hug25vtyi2wuru.ap-south-1.es.amazonaws.com"
OPENSEARCH_INDEX = "semantic-index"
opensearch = OpenSearch(
    hosts=[{"host": OPENSEARCH_ENDPOINT, "port": 443}],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
    timeout=60,
    max_retries=3,
    retry_on_timeout=True
)
ATHENA_DATABASE = "volty_curated_db_prod"
ATHENA_WORKGROUP = "primary"
ATHENA_OUTPUT_LOCATION = "s3://agent-athena-results/results/"

# ---------------------------------------------------------
# BEDROCK MODEL (using your inference profile, no guardrails for now)
# ---------------------------------------------------------
# Inference profile ARN:
# arn:aws:bedrock:ap-south-1:850920876043:inference-profile/apac.amazon.nova-lite-v1:0
# → model_id should be the profile ID part: "apac.amazon.nova-lite-v1:0"
bedrock_model = BedrockModel(
    model_id="apac.amazon.nova-lite-v1:0",
    streaming=True,
    #guardrail_id="epui8ex5py1o",   # Extracted from your ARN
    #guardrail_version="1",         # You confirmed version 1
    #guardrail_trace="enabled",
    temperature=0.3,
    max_tokens=10000,   # this is fine – this configures the Bedrock model itself
    region_name="ap-south-1"
)

def safe_text(x: str) -> str:
    """
    Escape characters that break Python f-strings:
    - { }
    - %
    """
    if not isinstance(x, str):
        x = str(x)
    return x.replace("{", "{{").replace("}", "}}").replace("%", "%%")


# Helper agent used *inside tools* instead of calling bedrock_model.invoke()
tool_llm_agent = Agent(
    name="tool_llm",
    model=bedrock_model,
    system_prompt=(
        "You are a backend utility model used by tools.\n"
        "- Always respond in exactly the format requested (JSON, SQL, XML, CSV, YAML, text, etc.).\n"
        "- Return ONLY the requested content with no explanations, no extra text, no formatting wrappers, and no code fences unless explicitly asked.\n"

    ),
)

# ---------------------------------------------------------
# Hook Provider (Logging)
# ---------------------------------------------------------
class LoggingHook(HookProvider):
    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(BeforeInvocationEvent, self.start)
        registry.add_callback(AfterInvocationEvent, self.end)

    def start(self, event: BeforeInvocationEvent) -> None:
        print(f"[START] {event.agent.name}")

    def end(self, event: AfterInvocationEvent) -> None:
        print(f"[END]   {event.agent.name}")


# ---------------------------------------------------------
# INTERNAL HELPERS (used by tools & agents)
# ---------------------------------------------------------
def _load_glue_schema() -> List[Dict[str, Any]]:
    """Read schema from Glue and convert to a compact form: [{name, columns: [...]}, ...]."""
    tables: List[Dict[str, Any]] = []

    paginator = glue.get_paginator("get_tables")
    pages = paginator.paginate(DatabaseName=ATHENA_DATABASE)

    for page in pages:
        for t in page["TableList"]:
            table_name = t["Name"]
            cols = t.get("StorageDescriptor", {}).get("Columns", [])
            part_cols = t.get("PartitionKeys", [])

            columns = [
                {"name": c["Name"], "type": c.get("Type", "")}
                for c in cols
            ] + [
                {"name": p["Name"], "type": p.get("Type", "")}
                for p in part_cols
            ]

            tables.append(
                {
                    "name": table_name,
                    "columns": columns,
                }
            )

    return tables

# ---------------------------------------------------------
# NEW: Reduce schema before sending to LLM (Option B)
# ---------------------------------------------------------
def _filter_relevant_tables(tables: List[Dict[str, Any]], user_query: str):
    """
    Reduce schema for LLM to prevent MaxTokensReachedException.
    - Always keep core tables.
    - Add other tables ONLY if mentioned in the user query.
    """

    CORE_TABLES = {"vehicle_data", "alerts", "agents", "cameras", "sites"}

    query = user_query.lower()
    filtered = []

    # Always keep core tables
    for t in tables:
        if t["name"] in CORE_TABLES:
            filtered.append(t)

    # Add additional tables ONLY if referenced in the query
    for t in tables:
        if t["name"] in CORE_TABLES:
            continue
        if t["name"] in query or t["name"].replace("_", " ") in query:
            filtered.append(t)

    # If nothing else matched → keep only core tables
    if not filtered:
        filtered = [t for t in tables if t["name"] in CORE_TABLES]

    # Remove duplicates
    filtered = list({t["name"]: t for t in filtered}.values())

    return filtered

def _smart_select_columns(columns: List[Dict[str, Any]]) -> List[str]:
    """
    Automatically detect important columns using patterns + basic schema intelligence.
    Input:  columns = [ { "name": "...", "type": "..." }, ... ]
    Output: list of column names.
    """
    KEY_PATTERNS = [
        "id", "ts", "timestamp", "date",
        "type", "status", "state",
        "speed", "count", "score",
        "site", "camera", "agent",
        "license", "plate", "region",
        "latitude", "longitude", "lat", "lon",
        "temperature", "heartbeat",
        "alert", "failure", "sync"
    ]

    # Names & types
    names = [c.get("name", "") for c in columns]
    types = {c.get("name", ""): c.get("type", "").lower() for c in columns}

    # 1) Pattern match columns by name
    pattern_cols = [
        n for n in names
        if any(p in n.lower() for p in KEY_PATTERNS)
    ]

    # 2) Promote numeric / measure-like columns by type
    numeric_types = ("int", "bigint", "double", "float", "decimal")
    numeric_cols = [
        n for n in names
        if any(t in types.get(n, "") for t in numeric_types)
    ]

    # 3) Promote primary-key-like columns
    pk_cols = [
        n for n in names
        if n.lower().endswith("_id") or n.lower() == "id"
    ]

    # 4) Promote timestamp-like columns
    ts_cols = [
        n for n in names
        if "ts" in n.lower() or "time" in n.lower() or "date" in n.lower()
    ]

    # Merge and dedupe with priority
    merged: List[str] = []
    for group in [pattern_cols, pk_cols, ts_cols, numeric_cols]:
        for n in group:
            if n and n not in merged:
                merged.append(n)

    # Fallback: if nothing selected, just take first few columns
    if not merged:
        merged = names[:5]

    # Cap at max 10 columns per table
    return merged[:10]


def _summarize_schema(tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build a compact schema for the LLM:
    [ { "name": "<table>", "columns": ["col1", "col2", ...] }, ... ]
    """
    summarized: List[Dict[str, Any]] = []

    for t in tables:
        important_cols = _smart_select_columns(t.get("columns", []))
        summarized.append(
            {
                "name": t["name"],
                "columns": important_cols,
            }
        )

    return summarized


def _compute_data_agent_plan(user_query: str) -> Dict[str, Any]:
    """
    Core logic for DataAgent:
    - Load Glue schema
    - Let LLM infer relevant tables/columns & filters from the question.
    """

    tables = _load_glue_schema()
    tables = _filter_relevant_tables(tables, user_query)
    # Remove misleading alert_type column from vehicle_data before passing to LLM
    for t in tables:
        if t["name"] == "vehicle_data":
            t["columns"] = [c for c in t["columns"] if c["name"] != "alert_type"]

    compact_schema = _summarize_schema(tables)
    safe_schema = safe_text(json.dumps(compact_schema))
    #minified_hints = " ".join(domain_hints.split())
    #safe_hints = safe_text(minified_hints)


    domain_hints ="""
DOMAIN KNOWLEDGE FOR ANPR SYSTEM (CRITICAL)

TABLE SUMMARIES (TRUE SOURCE OF COLUMNS — DO NOT USE ANY OTHERS):

alerts table:
Stores ANPR alerts / violations raised by agents.                                         
Columns:
    id                  → alert ID
    alert_type          → type of alert raised by the agent
    camera_id           → camera that captured the alert
    agent_id            → device that processed the alert
    site_id             → physical site where alert occurred
    confidence_score    → detection confidence (0–1)
    manual_review       → manual intervention flag

alerts.timestamp does NOT exist — use alerts.ts ONLY.
Only join alerts ↔ vehicle_data if explicitly required through license_plate.

agents table:
ANPR edge devices that receive camera streams and process vehicle captures.
Columns:
    agent_id
    agent_name
    agent_version
    hardware_version
    site_id
    status
    temperature
    host_name
    http_anpr_url
    http_health_url
    http_max_tries
    http_que_size
    motion_threshold
    last_retrewied_date  → ETL heartbeat timestamp (ONLY for active/inactive)
    health_check_interval
    heartbeat_interval

    # FOTA (Firmware OTA)
    fota_ip
    fota_port
    fota_user
    fota_passcode
    fota_check_interval
    fota_max_retries
    fota_retry_interval

    # FTP (Image / data upload)
    ftp_ip
    ftp_port
    ftp_user
    ftp_passcode
    ftp_que_size
    ftp_retry_delay
    max_file_size

    # MQTT (publish/subscribe control & events)
    mqtt_broker
    mqtt_port
    mqtt_topic_pub
    mqtt_topic_sub

Agent activity MUST use ONLY ts:
      ACTIVE   → ts >= now() - interval '10' minute
      INACTIVE → ts <  now() - interval '10' minute
DO NOT use status or op_type for active/inactive.

cameras table:
Metadata + operational settings for cameras connected to agents.
Columns:
    camera_id
    camera_name
    camera_position
    agent_id
    site_id
    frame_rate
    resolution
    mode
    anti_flicker
    op_type

resolution is STRING (e.g., "720p", "1080p", "4k")
  → DO NOT apply numeric comparisons such as > or <.

sites table:
Physical deployment locations.
Columns:
    site_id
    site_name
    site_address
    ts  → site event timestamp

failed_sync_records table:
Logs of failed ingest attempts from agents.
Columns:
    id
    failed_at        → timestamp of failure
    error_message
    op_type

NEVER use record_data_struct fields for analytics.

vehicle_data table:
Every vehicle captured at camera level.
Columns:
    id
    license_plate
    color
    plate_colour
    description
    model
    make
    vehicle_type
    speed
    direction
    description
    region
    agent_id
    camera_id
    site_id
    plate_img_url
    vehicle_img_url
    video
    latitude
    longitude
vehicle_data DOES NOT contain confidence_score.
alert_type in vehicle_data is NOT the same as alerts.alert_type (ignore it).

olake / injection metadata:
injection_id        → ingestion batch ID (CDC)
olake_timestamp     → timestamp of CDC ingestion  
NOT allowed for analytics (same rule as last_retrewied_date).

---------------------------------------------------------------
RELATIONSHIP RULES (STRICT — USE ONLY THESE):

vehicle_data.camera_id = cameras.camera_id  
alerts.camera_id       = cameras.camera_id  
vehicle_data.site_id   = sites.site_id  
alerts.site_id         = sites.site_id  
agents.site_id         = sites.site_id  
vehicle_data.agent_id  = agents.agent_id  
alerts.agent_id        = agents.agent_id  

alerts.license_plate ↔ vehicle_data.license_plate ONLY when explicitly required.

INVALID JOINS:
vehicle_data ↔ alerts (without license_plate)
failed_sync_records ↔ any analytic table
olake/injection metadata ↔ analytic tables

---------------------------------------------------------------
TIMESTAMP RULES (CRITICAL):

Allowed analytic timestamps:
    alerts.ts
    vehicle_data.ts
    sites.ts
    agents.last_retrewied_date  (ONLY for activity)
    cameras.last_retrewied_date (ONLY for activity)
    failed_sync_records.failed_at

Do NOT use:
    olake_timestamp
    injection_id timestamps
    last_retrewied_date for analytics

Valid Athena formats:
    TIMESTAMP 'YYYY-MM-DD HH:MM:SS'
    'YYYY-MM-DD HH:MM:SS'
NO ISO formats (e.g., '2024-01-31T10:05:00').

---------------------------------------------------------------
NATURAL LANGUAGE → TABLE MAPPING:

“traffic”, “speed”, “vehicle counts”, “busy areas” → vehicle_data  
“alerts”, “violations”, “spikes”, “confidence”     → alerts  
“agent health”, “inactive agents”                → agents  
“camera performance”, “resolution”, “FPS”        → cameras  
“site activity”, “locations”                     → sites  
“sync errors”, “pipeline failures”               → failed_sync_records  

---------------------------------------------------------------
NATURAL LANGUAGE → FILTER MAPPING (STRICT):

"inactive agents" → agents.ts <  now() - interval '10' minute  
"active agents"   → agents.ts >= now() - interval '10' minute  

Time filters MUST use the table’s native ts:
    today        → date(ts) = current_date  
    yesterday    → date(ts) = current_date - interval '1' day  
    last hour    → ts >= now() - interval '1' hour  
    last 24 hours → ts >= now() - interval '24' hour  

CRITICAL RULE:
If timeframe is not mentioned → DO NOT apply any ts filter.

If timeframe is given but table has zero rows → return 0 rows.  
DO NOT fall back or expand window.

---------------------------------------------------------------
ABSOLUTE RESTRICTIONS:
Do NOT invent columns.
Do NOT use alerts.timestamp (invalid).
Do NOT join unrelated tables.
Do NOT use last_retrewied_date or olake_timestamp for analytics.
Do 

NOT apply numeric comparisons to cameras.resolution."""
    
    # --- Minify and escape domain hints to avoid brace conflicts ---
    minified_hints = " ".join(domain_hints.split())
    safe_hints = safe_text(minified_hints)


    # --- Build the DataAgent planning prompt ---
    # --- Build DataAgent prompt safely (NO f-strings) ---
    prompt = (
        "You are DataAgent — a schema-aware, context-aware planner that interprets "
        "the user query using the Glue schema and ANPR domain rules.\n\n"
        "User question:\n" +
        safe_text(user_query) + "\n\n"
        "Glue Catalog schema (summarized):\n" +
        safe_schema + "\n\n"
        "Domain rules (condensed):\n" +
        safe_hints + "\n\n"
        "YOUR TASK:\n"
        "- Understand the user’s intent using context, keywords, semantics, and domain knowledge.\n"
        "- Identify the smallest set of tables required to answer the question.\n"
        "- Select ONLY real columns (never invent).\n"
        "- Infer optional WHERE-style filters using numeric, boolean, and time logic.\n"
        "- Use ONLY valid timestamp columns.\n"
        "- If query is vague, return empty filters.\n"
        "- Respect only the valid joins defined in rules.\n\n"
        "OUTPUT FORMAT (STRICT JSON ONLY):\n"
        "{\n"
        "  \"tables\": [\n"
        "    {\n"
        "      \"name\": \"<table_name>\",\n"
        "      \"reason\": \"<why this table is relevant>\",\n"
        "      \"columns\": [\"col1\", \"col2\"]\n"
        "    }\n"
        "  ],\n"
        "  \"filters\": [\"<filter>\"],\n"
        "  \"reasoning\": \"<brief explanation>\"\n"
        "}\n\n"
        "RULES:\n"
        "- JSON only. No markdown or SQL.\n"
        "- No hidden reasoning.\n"
        "- No invented columns or tables.\n"
    )
    resp = tool_llm_agent(prompt)
    text = str(resp).strip()

# --- FIX: REMOVE code fences for valid JSON ----
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
            # Additional cleanup (model sometimes adds trailing ``` accidentally)
            if text.endswith("```"):
                text = text[:-3].strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = {
            "tables": [],
            "filters": [],
            "reasoning": f"LLM did not return valid JSON. Raw: {text[:500]}",
            }

    return {
        "plan": parsed,
        "schema": compact_schema, 
        }

 
# ---------------------------------------------------------
# DATA AGENT TOOL: Read schema directly from Glue Catalog
# ---------------------------------------------------------
@tool
def data_agent_plan(user_query: str) -> Dict[str, Any]:
    """
    DataAgent tool:
    - Reads table & column schema from AWS Glue Catalog
    - Maps user intent to relevant tables/columns

    Returns:
    {
      "plan": { ... LLM interpretation ... },
      "schema": [ { "name": "...", "columns": [...] }, ... ]
    }
    """
    return _compute_data_agent_plan(user_query)


# ---------------------------------------------------------
# VECTOR AGENT TOOL (OpenSearch)
# ---------------------------------------------------------
@tool
def vector_agent_search(
    user_query: str,
    plan: Optional[Dict[str, Any]] = None,
    top_k: int = 5
) -> Dict[str, Any]:

    # -------------------------------------------
    # Extract table names safely
    # -------------------------------------------
    try:
        table_list = plan.get("tables", []) if plan else []
        table_names = [t.get("name") for t in table_list if isinstance(t, dict)]
    except Exception:
        table_names = []

    # -------------------------------------------
    # Rewrite query using LLM
    # -------------------------------------------
    rewrite_prompt = (
    "Rewrite the user query for semantic retrieval.\n"
    "Keep only essential meaning.\n"
    "Remove noise and ambiguity.\n\n"
    "User query:\n" +
    safe_text(user_query) +
    "\n\nReturn ONLY the rewritten query."
    )

    rewritten_query = str(tool_llm_agent(rewrite_prompt)).strip()

    # -------------------------------------------
    # Build OpenSearch query
    # -------------------------------------------
    vector_query = {
        "size": top_k,
        "query": {
            "multi_match": {
                "query": rewritten_query,
                "type": "best_fields",
                "fields": ["text^3", "metadata.*^1"]
            }
        }
    }

    # -------------------------------------------
    # Run search safely
    # -------------------------------------------
    try:
        results = opensearch.search(index=OPENSEARCH_INDEX, body=vector_query)
        cleaned_hits = []

        for h in results["hits"]["hits"][:5]:  # limit to top 3 hits
            src = h.get("_source", {})

            # ---- Remove embedding field completely ----
            if "embedding" in src:
                src.pop("embedding", None)

            # ---- Clean text ----
            text = src.get("text", "")
            if isinstance(text, str):
                text = (
                    text.replace("\n", " ")
                        .replace("\t", " ")
                        .strip()
                )
                if len(text) > 350:
                    text = text[:350] + " …[TRUNCATED]"

            # ---- Clean metadata ----
            # ---- Clean metadata ----
            metadata = src.get("metadata", {})
            safe_meta = {}

            def scrub(value):
                # Remove arrays (embeddings)
                if isinstance(value, list):
                    return None
                if isinstance(value, str):
                    v = value.replace("\n", " ").replace("\t", " ").strip()
                    if len(v) > 120:
                        v = v[:120] + " …[TRUNCATED]"
                    return v
                return value

            # Clean metadata safely
            # Clean metadata first
            if isinstance(metadata, dict):
                for meta_key, meta_value in metadata.items():
                    if meta_key.lower() in ["embedding", "vector", "vec", "emb"]:
                        continue
                    if isinstance(meta_value, dict) and "embedding" in meta_value:
                        meta_value.pop("embedding", None)
                    safe_meta[meta_key] = scrub(meta_value)

            # Now detect table
            possible_table_keys = [
                            "table",
                            "table_name",
                            "source_table",
                            "glue_table",
                            "opensearch_table",
                            "os_table",
                            "index",
                            "index_name",
                            "dataset"
                        ]

            # Detect table name from metadata
            table_name = "unknown"

            if isinstance(metadata, dict):
                for key in possible_table_keys:
                    if key in metadata:
                        table_name = metadata[key]
                        break


            cleaned_hits.append({
                "score": float(h.get("_score", 0)),
                "text": text,
                "metadata": safe_meta,
                "table": table_name

            })

        return {
            "rewritten_query": rewritten_query,
            "hits": cleaned_hits,
            "raw_plan": plan,
        }

    except Exception as e:
        logger.error(f"Vector search error: {e}")
        return {
            "rewritten_query": rewritten_query,
            "hits": [],
            "raw_plan": plan,
            "error": str(e)
        }
# ---------------------------------------------------------
# ATHENA AGENT TOOL
# ---------------------------------------------------------
def _poll_athena(qid: str) -> Dict[str, Any]:
    """Helper to wait for Athena query to complete."""
    while True:
        status = athena.get_query_execution(QueryExecutionId=qid)
        state = status["QueryExecution"]["Status"]["State"]
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            return status
        time.sleep(2)


def _fetch_results(qid: str):
    """Fetch Athena query results into (columns, rows)."""
    paginator = athena.get_paginator("get_query_results")
    pages = paginator.paginate(QueryExecutionId=qid)

    rows: List[Dict[str, Any]] = []
    cols: List[str] = []
    first = True

    for page in pages:
        result = page["ResultSet"]
        cols = [c["Name"] for c in result["ResultSetMetadata"]["ColumnInfo"]]

        if first:
            data_rows = result["Rows"][1:]  # skip header
            first = False
        else:
            data_rows = result["Rows"]

        for r in data_rows:
            vals = [c.get("VarCharValue") for c in r["Data"]]
            rows.append(dict(zip(cols, vals)))

    return cols, rows


@tool
def athena_agent_query(
    user_query: str,
    plan: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    AthenaAgent:
    - Generates SQL using the LLM and Glue-aware plan
    - Executes Athena query
    - Returns rows + SQL used

    If `plan` is None, it will internally call DataAgent logic to infer it.
    """

    # Fallback: self-plan if planner forgot to pass plan
    internal_plan = None
    if plan is None:
        internal_plan = _compute_data_agent_plan(user_query)
        plan = internal_plan["plan"]

    prompt = (
    "You are AthenaSQL — a highly accurate SQL generation engine specialized in Amazon Athena "
    "(Trino/Presto SQL dialect, ANSI compliant).\n\n"

    "Your role:\n"
    "- Convert the user’s natural-language question into a correct Athena SQL query.\n"
    "- Use ONLY the tables, columns, and filters provided in:\n"
    "    • the DataAgent + Planner plan\n"
    "    • the Glue Catalog schema\n"
    "- NEVER hallucinate tables, columns, JOIN keys, conditions, or timestamps.\n\n"

    "User question:\n" +
    safe_text(user_query) + "\n\n"

    "Tables/filter plan (from DataAgent + Planner):\n" +
    safe_text(json.dumps(plan, indent=2)) + "\n\n"

    "Warehouse details:\n"
    f"- Database name: {ATHENA_DATABASE}\n"
    "- Athena uses Trino SQL syntax (NOT PostgreSQL).\n"
    "Valid examples:\n"
    "    • date_trunc('month', current_date)\n"
    "    • now()\n"
    "    • current_timestamp\n"
    "    • interval '1' day, interval '24' hour\n"
    "    • date(ts) = current_date\n"
    "- TIMESTAMP-like fields include: ts, last_retrewied_date, last_sync_ts, failed_at, olake_timestamp.\n"
    "ALERTS TABLE TIMESTAMP RULE:\n"
    "- The alerts table does NOT have a column named \"timestamp\".\n"
    "- The ONLY valid timestamp column in alerts is \"ts\".\n"
    "- SQL MUST NOT generate \"alerts.timestamp\".\n\n"

    "----------------------------------------------------------------------\n"
    "CRITICAL SQL JOIN SAFETY RULES (MUST FOLLOW)\n"
    "----------------------------------------------------------------------\n"
    "- NEVER join aggregations that do not share a real relational key.\n"
    "- Valid relationships ONLY:\n"
    "      vehicle_data.camera_id = cameras.camera_id\n"
    "      cameras.site_id = sites.site_id\n"
    "      vehicle_data.agent_id = agents.agent_id\n"
    "- alerts MAY be grouped by license_plate but MUST NOT be joined to:\n"
    "      • vehicle_traffic aggregates\n"
    "      • camera/site aggregates\n"
    "      • any table without a matching relational key\n"
    "- NEVER join vehicle_traffic with vehicle_alerts.\n"
    "----------------------------------------------------------------------\n\n"

    "Time-window interpretation:\n"
    "today → date(<ts>) = current_date\n"
    "yesterday → date(<ts>) = current_date - interval '1' day\n"
    "last 24 hours → <ts> >= now() - interval '24' hour\n"
    "last 7 days → <ts> >= current_date - interval '7' day\n"
    "last month:\n"
    "    <ts> >= date_trunc('month', current_date - interval '1' month)\n"
    "    AND <ts> < date_trunc('month', current_date)\n\n"

    "SQL construction behavior:\n"
    "- Infer JOINs only when valid relationships exist AND appear in the plan.\n"
    "- Apply all filters exactly as provided.\n"
    "- Prefer explicit column selection.\n"
    "- Avoid SELECT * unless necessary.\n"
    "- Use GROUP BY / ORDER BY / LIMIT only when required.\n\n"

    "Requirements:\n"
    "- Use ONLY tables + columns identified in the DataAgent/Planner plan.\n"
    "- Generate SQL that is valid Athena SQL.\n"
    "- Do NOT add commentary, markdown, JSON, or backticks.\n\n"

    "Output rules:\n"
    "- Return ONLY the raw SQL text.\n"
    "- No markdown.\n"
    "- No comments.\n"
    "- No explanation.\n"
)

    sql_resp = tool_llm_agent(prompt)
    # --- FIX: Auto-correct invalid "timestamp" column in alerts ---
    # --- FIX: Remove invalid vehicle_data.alert_type references ---
    sql = str(sql_resp).strip()

    # --- FIX: Auto-correct invalid "timestamp" column in alerts ---
    if "volty_curated_db_prod.alerts" in sql:
        sql = sql.replace(" alerts.\"timestamp\"", " alerts.ts")
        sql = sql.replace("alerts . timestamp", "alerts.ts")
        sql = sql.replace("alerts. timestamp", "alerts.ts")
        sql = sql.replace("alerts.timestamp", "alerts.ts")
        sql = sql.replace("alert.timestamp", "alerts.ts")


    # --- FIX: Remove invalid vehicle_data.alert_type references ---
    sql = sql.replace("vehicle_data.alert_type", "")
    sql = sql.replace("vd.alert_type", "")


    # Just in case the model accidentally returned fenced code
    if sql.startswith("```"):
        sql = sql.strip("`").strip()
        if "\n" in sql:
            sql = sql.split("\n", 1)[1].strip()

    logger.info("AthenaAgent generated SQL:\n%s", sql)

    try:
        start = athena.start_query_execution(
            QueryString=sql,
            QueryExecutionContext={"Database": ATHENA_DATABASE},
            ResultConfiguration={"OutputLocation": ATHENA_OUTPUT_LOCATION},
            WorkGroup=ATHENA_WORKGROUP,
        )
        qid = start["QueryExecutionId"]
        status = _poll_athena(qid)

        if status["QueryExecution"]["Status"]["State"] != "SUCCEEDED":
            return {
                "sql": sql,
                "error": f"Athena query failed with state {status['QueryExecution']['Status']['State']}",
                "internal_plan": internal_plan,
            }

        cols, rows = _fetch_results(qid)
        return {
            "sql": sql,
            "columns": cols,
            "rows": rows,
            "internal_plan": internal_plan,
        }

    except Exception as e:
        logger.exception(" Athena query exception ")
        return {
            "sql": sql,
            "error": str(e),
            "internal_plan": internal_plan,
        }


# ---------------------------------------------------------
# PLANNER AGENT (main orchestrator)
# ---------------------------------------------------------
planner_agent = Agent(
    name="planner",
    model=bedrock_model,
    tools=[data_agent_plan, vector_agent_search, athena_agent_query],
    tool_executor=ConcurrentToolExecutor(),
    hooks=[LoggingHook()],
    state={"planner_invocations": 0, "max_invocations": 1},
    system_prompt="""
        You are PlannerAgent — an intelligent orchestrator that selects the correct toolchain based on the user query, the Glue Catalog schema, and the DataAgent plan.

        You MUST:
        1. ALWAYS call `data_agent_plan` FIRST.
        - This step provides schema interpretation, relevant tables, inferred filters, and domain signals such as timestamps, agent activity, heartbeat intervals, status flags, and failure indicators.

        2. After receiving the DataAgent plan, you MUST intelligently decide which tool(s) to call next using deep semantic understanding, schema awareness, and domain inference.

        ================================================================================
        INTELLIGENT ROUTING LOGIC
        ================================================================================

        Call `vector_agent_search` when:
        - The user query is vague, incomplete, or ambiguous.
        - The query is conceptual or asks for explanations (e.g., "what is", "explain", "describe", "meaning of").
        - The question is natural-language heavy rather than analytical.
        - User asks for documentation-style or descriptive information.
        - The DataAgent plan shows weak or low-confidence table selection.
        - The domain request is about understanding *concepts* rather than retrieving *metrics*.

        Examples:
        - “Explain how agent heartbeat works.”
        - “What does inactive mean?”
        - “Describe the agent lifecycle.”
        - “How do failures happen?”

        Call `athena_agent_query` when:
        - The user asks for metrics, counts, KPIs, lists, filters, or aggregations.
        - The user refers to timestamps such as ts, last_sync_ts, olake_timestamp, failed_at, last_retrewied_date.
        - The user requests operational insights.
        - The DataAgent plan identifies clear tables + filters.
        - The query implies analytics or structured data retrieval.

        Examples:
        - “List inactive agents.”
        - “Show failed jobs in the last 24 hours.”
        - “Count agents with stale heartbeats.”
        - “Give me the latest heartbeat times.”

        Call BOTH tools when:
        - The user mixes conceptual description + analytical request.
        Example:
        “Explain what inactive agents are and list them.”

        In such cases:
        1. Call `vector_agent_search` FIRST.
        2. Then call `athena_agent_query` for structured results.

        ================================================================================
        CRITICAL SAFETY RULES (MUST FOLLOW)
        ================================================================================

        ERROR HANDLING & STOP CONDITIONS:
        - If ANY tool returns an "error" field:
        - If the Athena error indicates an invalid JOIN, you MUST:
            * Remove all JOINs between unrelated tables.
            * Split the query into separate SELECT statements.
            * Retry athena_agent_query once with corrected instructions.
            * The Planner MUST analyze the error and understand it.
            * If the error is FIXABLE (SQL syntax, missing column, bad filter, malformed JSON):
                → Modify the tool input and retry.
            * If the error is NOT fixable:
                → STOP immediately and return Planner JSON.

        CONTROLLED RETRY POLICY:
        - You may call the same tool a MAXIMUM of 3 times (1 initial + 2 retries).
        - Each retry MUST:
            * Modify the tool input meaningfully.
            * Fix the specific issue shown in the error.
            * NEVER repeat the exact same call.
        - After 3 failed attempts:
            * STOP.
            * Return Planner JSON with the last error included.

        PREVENT INFINITE LOOPS:
        - NEVER call a tool more than 3 times.
        - NEVER repeatedly call `athena_agent_query` with the same SQL.
        - NEVER loop DataAgent if it returns invalid JSON repeatedly.
        - NEVER retry a tool unless the input has been corrected.

        MANDATORY STEP LOGGING:
        - Every tool call MUST be added to the "steps" array.
        - If a tool error occurs:
            * Include that error in "steps".
            * Retry ONLY if actionable and within the 3-attempt limit.
            * Otherwise STOP and return Planner JSON.

        These safety rules OVERRIDE all other behaviors.

        ================================================================================
        OPTIONAL IMPROVEMENT (ENABLED)
        ================================================================================

        Use the following natural-language signals:

        Vague / Semantic Indicators → Prefer vector search:
        - “explain”, “what is”, “why”, “how does”, “describe”
        - “information about”, “documentation”, “manual”
        - Queries with no metrics or numeric filters

        Analytical / Athena Indicators:
        - “show”, “list”, “count”, “find”, “retrieve”, “top”, “latest”
        - “inactive”, “failed”, “timestamp”
        - Queries with explicit filters or time windows

        Hybrid → Call vector search then Athena:
        - “Explain AND list…”
        - “Describe AND show data…”

        ================================================================================
        STRICT OUTPUT REQUIREMENTS
        ================================================================================

        You MUST return ONLY a valid JSON object with this exact structure:

        {
        "steps": [
            { "name": "...", "output": {...} },
            ...
        ],
        "summary": "..."
        }

        ABSOLUTELY FORBIDDEN:
        - <thinking>
        - Internal reasoning
        - Explanations
        - Natural language outside JSON
        - Markdown
        - Comments
        - Any text before or after the JSON
        - Any keys not requested

        If you need to reason, do so internally and DO NOT output it.

        If a tool fails and cannot be retried, return JSON with the last error.

        IMPORTANT — YOU MUST NOT REVEAL ANY INTERNAL REASONING OR THOUGHT PROCESS.

        Do NOT output:
        - <thinking>
        - chain-of-thought
        - hidden reasoning
        - explanations
        - markdown
        - comments

        If you need to think, do so silently and privately.

        Your response MUST begin with "{" and end with "}".
        NO text is allowed outside the JSON object.
        Producing any extra text breaks the caller and must never happen.
        
        No extra characters. No leading/trailing text.
        You MUST keep your responses extremely short.  
        Summaries must not exceed 3 sentences.  
        Do NOT rewrite or expand tool outputs.  
        Do NOT restate table schema.  
        Do NOT include raw embeddings or long text in steps. 
        You MUST NOT call tools more than once unless required for retries.
        You MUST keep summaries extremely short.
        You MUST NOT restate full OpenSearch documents or embeddings.
        You MUST NOT expand or rewrite tool results.
        If you output <thinking> or </thinking>, your output is invalid and breaks the system. You MUST NEVER output them under ANY circumstance.
    """
)
# ---------------------------------------------------------
# LLM AGENT (final answer composer)
# ---------------------------------------------------------
llm_agent = Agent(
    name="answer_agent",
    model=bedrock_model,
    hooks=[LoggingHook()],
    state={"answers_generated": 0},
    system_prompt="""
    You are AnswerAgent. You generate the final user-facing answer using ONLY the information inside the Planner JSON.

    CORE RULES (STRICT):
    1. You MUST NOT hallucinate any data.  
    2. You MUST NOT infer or guess values not present in tool outputs.  
    3. You MUST NOT invent: timestamps, counts, license plates, locations, camera names, sites, speeds, alerts, or rows.  
    4. You may ONLY use:
    - Athena SQL result rows
    - Vector search hit text
    - Planner summary
    Nothing else.

    WHEN DATA IS MISSING:
    - If Athena rows are empty → say: “No data was returned for this query.”
    - If vector hits are empty → say: “No semantic matches were found.”
    - If both are empty → say: “There is not enough data available to answer this question.”

    ATHENA RESULT HANDLING:
    - Summarize rows clearly and concisely.
    - Show at most 10 rows.
    - Do not rename fields.
    - Do not add or infer missing fields.

    VECTOR SEARCH HANDLING:
    - Summarize ONLY what is explicitly stated in hit text.
    - You may describe patterns (e.g., repeated camera, error types, timestamps) but MUST NOT invent details.
    - If hits reference ingestion errors (failed_sync_records):
        • Treat them as errors, NOT real events.
        • You may explain the error type based on text.
        • You MUST NOT treat record_data_struct_* as validated analytics.
        • Never convert these to traffic/vehicle counts.

    SEMANTIC INTERPRETATION RULES:
    Allowed → describing trends directly visible in hits  
    Not allowed → adding new numbers, plates, timestamps, or events  

    PRIORITY ORDER:
    1. Use Athena results if they exist.
    2. Then enrich with any relevant vector insights.
    3. If the query is conceptual (no Athena data expected), rely only on vector hits.

    FORBIDDEN OUTPUT:
    - JSON
    - Code
    - SQL
    - Tool explanations
    - Internal reasoning
    - Mentions of prompts, schema, or system rules
    If you output <thinking> or </thinking>, your output is invalid and breaks the system. You MUST NEVER output them under ANY circumstance.
    Your response must be a clean, concise natural language answer tailored to the user query.
"""
)

# ---------------------------------------------------------
# MAIN CLI
# ---------------------------------------------------------
def main():
    q = input("Enter your analytics question: ")

    # 1) Run planner (tool orchestration)
    planner_resp = planner_agent(q)
    planner_raw = str(planner_resp).strip()  # AgentResult → string via __str__

        # --- CLEAN PLANNER OUTPUT ---
    cleaned = planner_raw.strip()
    cleaned = cleaned.replace("<thinking>", "").replace("</thinking>", "")
    # Remove any text before the starting '{'
    if "{" in cleaned:
        first = cleaned.find("{")
        last = cleaned.rfind("}")
        if first != -1 and last != -1:
            cleaned = cleaned[first:last+1]
        else:
            cleaned = "{}"


    try:
        cleaned = cleaned[: cleaned.rfind("}") + 1]
        planner_json = json.loads(cleaned)
    except Exception:
        planner_json = {"error": "Planner returned invalid JSON", "raw": cleaned}
    safe_planner = {
    "steps": planner_json.get("steps", []),
    "summary": planner_json.get("summary")
    }
    if len(safe_planner["steps"]) > 4:
        safe_planner["steps"] = safe_planner["steps"][-4:]
    # 2) Ask answer_agent to write final response
    final_prompt = f"""
    User Query:
    {q}
    Planner JSON:
    {json.dumps(safe_planner, indent=2)}
    Write the final answer for the user.
"""
    answer_resp = llm_agent(final_prompt)
    answer_text = str(answer_resp).strip()

    print("\n===== FINAL ANSWER =====")
    print(answer_text)

# ---------------------------------------------------------
# AWS LAMBDA HANDLER ENTRYPOINT
# ---------------------------------------------------------
def handler(event, context):
    """
    AWS Lambda handler.
    Expects event = { "prompt": "<user question>" }
    """

    user_query = event.get("prompt")
    if not user_query:
        return {
            "error": "Missing 'prompt' in event payload",
            "example": {"prompt": "List inactive agents"}
        }

    # 1) Run PlannerAgent
    planner_resp = planner_agent(user_query)
    planner_raw = str(planner_resp).strip()
    # --- CLEAN PLANNER OUTPUT ---
    cleaned = planner_raw.strip()
    cleaned = cleaned.replace("<thinking>", "").replace("</thinking>", "")
    if "{" in cleaned:
        first = cleaned.find("{")
        last = cleaned.rfind("}")
        if first != -1 and last != -1:
            cleaned = cleaned[first:last+1]
        else:
            cleaned = "{}"


    try:
        planner_json = json.loads(cleaned)
    except Exception:
        planner_json = {"error": "Planner returned invalid JSON", "raw": cleaned}
    safe_planner = {
    "steps": planner_json.get("steps", []),
    "summary": planner_json.get("summary")
    }
    # --- FIX: Prevent Planner from returning massive step history ---
    if len(safe_planner["steps"]) > 4:
        safe_planner["steps"] = safe_planner["steps"][-4:]

    # 2) Run AnswerAgent
    final_prompt = f"""
    User Query:
    {user_query}

    Planner JSON:
    {json.dumps(safe_planner, indent=2)}

    Write the final answer for the user.
"""
    answer_resp = llm_agent(final_prompt)
    answer_text = str(answer_resp).strip()

    return {
        "answer": answer_text,
        "planner_steps": planner_json
    }

if __name__ == "__main__":
    main()