# querygpt_agent.py

import sqlite3
import os
import json
import logging
import re
import random
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, TypedDict

# New import for async SQLite
import aiosqlite

# SQL parsing
from sqlglot import parse_one, exp
from sqlglot.errors import ParseError

# LangGraph imports
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# OpenAI imports (async client)
from openai import AsyncOpenAI, APIError, RateLimitError

# Observability imports
# pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http langchain-core
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.trace import set_tracer_provider

# CORRECTED IMPORT FOR OPENTELEMETRY CALLBACK HANDLER
from langchain_core.tracers.langchain import LangChainTracer
OpenTelemetryCallbackHandler = LangChainTracer

# === CRITICAL CHANGE: REMOVED nest_asyncio.apply() ===
# Uvicorn (FastAPI server) handles the asyncio event loop.

# Configure logging to be structured (JSON format for better parsing in production)
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "file": f"{record.filename}:{record.lineno}",
            "function": record.funcName,
            "name": record.name,
        }
        if record.exc_info:
            log_record["exc_info"] = self.formatException(record.exc_info)

        span_context = trace.get_current_span().get_span_context()
        if span_context.is_valid:
            log_record["trace_id"] = f"{span_context.trace_id:x}"
            log_record["span_id"] = f"{span_context.span_id:x}"

        return json.dumps(log_record)

# Remove existing handlers to avoid duplicates and ensure our custom formatter is used
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Set up logging with our JSON formatter
handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler])

# --- OpenTelemetry Setup ---
def setup_opentelemetry():
    resource = Resource.create({"service.name": "QueryGPT"})
    provider = TracerProvider(resource=resource)
    processor = SimpleSpanProcessor(ConsoleSpanExporter()) # Exports traces to console
    provider.add_span_processor(processor)
    set_tracer_provider(provider)
    logging.info("OpenTelemetry configured with ConsoleSpanExporter.")

# Initialize OpenTelemetry (call once at script start)
setup_opentelemetry()

# --- Shared Database Setup (from previous versions) ---
DB_NAME = 'uber_analytics.db'

async def setup_database_if_not_exists(): # Now an async function
    """Sets up the database and populates it with sample data if it doesn't exist."""
    if os.path.exists(DB_NAME):
        logging.info(f"Database '{DB_NAME}' already exists. Skipping setup.")
        return

    logging.info(f"Setting up database '{DB_NAME}'...")
    async with aiosqlite.connect(DB_NAME) as conn:
        cursor = await conn.cursor()

        await cursor.execute("""
            CREATE TABLE uber_drivers (
                driver_id TEXT PRIMARY KEY, driver_name TEXT, city TEXT,
                rating REAL, status TEXT, signup_date DATE
            )
        """)
        await cursor.execute("""
            CREATE TABLE uber_users (
                user_id TEXT PRIMARY KEY, user_name TEXT, signup_date DATE,
                preferred_payment_method TEXT
            )
        """)
        await cursor.execute("""
            CREATE TABLE uber_cities (
                city_id TEXT PRIMARY KEY, name_of_city TEXT, is_operational BOOLEAN,
                market TEXT, country TEXT, ctry_iso2 TEXT, tz TEXT
            )
        """)
        await cursor.execute("""
            CREATE TABLE uber_trips_data (
                trip_id TEXT PRIMARY KEY, user_id TEXT, driver_id TEXT,
                vehicle_type TEXT, trip_start_location TEXT, trip_end_location TEXT,
                trip_distance_miles REAL, trip_fare_usd REAL, surge_multiplier REAL,
                trip_status TEXT, request_timestamp DATETIME, start_timestamp DATETIME,
                end_timestamp DATETIME, is_trip_completed BOOLEAN, date_val DATE,
                FOREIGN KEY (user_id) REFERENCES uber_users(user_id),
                FOREIGN KEY (driver_id) REFERENCES uber_drivers(driver_id)
            )
        """)
        await cursor.execute("""
            CREATE TABLE uber_payments (
                payment_id TEXT PRIMARY KEY, trip_id TEXT, amount_usd REAL,
                payment_method TEXT, payment_timestamp DATETIME, status TEXT,
                FOREIGN KEY (trip_id) REFERENCES uber_trips_data(trip_id)
            )
        """)
        await conn.commit()

        # Sample Data Generation
        cities_data = []
        cities_list = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Miami', 'Seattle', 'Boston', 'San Francisco']
        markets = ['US_East', 'US_West', 'US_Central']
        for i, city_name in enumerate(cities_list):
            city_id = f'CITY{i+1:03d}'
            is_operational = True
            market = random.choice(markets)
            country = 'USA'
            ctry_iso2 = 'US'
            tz = 'America/New_York' if 'New York' in city_name else 'America/Los_Angeles' if 'Los Angeles' in city_name else 'America/Chicago'
            cities_data.append((city_id, city_name, is_operational, market, country, ctry_iso2, tz))
        await cursor.executemany("INSERT INTO uber_cities VALUES (?, ?, ?, ?, ?, ?, ?)", cities_data)

        drivers_data = []
        statuses = ['active', 'inactive', 'on_vacation']
        for i in range(1, 21):
            driver_id = f'DRV{i:03d}'
            driver_name = f'Driver Name {i}'
            city = random.choice(cities_list)
            rating = round(random.uniform(3.0, 5.0), 1)
            status = random.choice(statuses)
            signup_date = (datetime.now() - timedelta(days=random.randint(30, 365))).strftime('%Y-%m-%d')
            drivers_data.append((driver_id, driver_name, city, rating, status, signup_date))
        await cursor.executemany("INSERT INTO uber_drivers VALUES (?, ?, ?, ?, ?, ?)", drivers_data)

        users_data = []
        payment_methods = ['credit_card', 'paypal', 'apple_pay']
        for i in range(1, 51):
            user_id = f'USR{i:03d}'
            user_name = f'User Name {i}'
            signup_date = (datetime.now() - timedelta(days=random.randint(15, 300))).strftime('%Y-%m-%d')
            preferred_payment_method = random.choice(payment_methods)
            users_data.append((user_id, user_name, signup_date, preferred_payment_method))
        await cursor.executemany("INSERT INTO uber_users VALUES (?, ?, ?, ?)", users_data)

        trips_data = []
        payments_data = []
        trip_statuses = ['completed', 'cancelled', 'in_progress']
        all_driver_ids = [d[0] for d in drivers_data]
        all_user_ids = [u[0] for u in users_data]
        locations = ['Midtown', 'Downtown', 'Brooklyn', 'Queens', 'Harlem', 'Seattle', 'Chicago', 'New York', 'Los Angeles', 'Houston']

        vehicle_types = ['Sedan', 'SUV', 'Tesla', 'Luxury']

        for i in range(1, 201): # 200 trips
            trip_id = f'TRP{i:04d}'
            user_id = random.choice(all_user_ids)
            driver_id = random.choice(all_driver_ids)
            start_loc = random.choice(locations)
            end_loc = random.choice(locations)
            while end_loc == start_loc:
                end_loc = random.choice(locations)

            vehicle_type = random.choice(vehicle_types)
            trip_distance = round(random.uniform(1.0, 25.0), 1)
            trip_fare = round(random.uniform(5.0, 75.0), 2)
            surge_multiplier = round(random.uniform(1.0, 2.5) if random.random() < 0.3 else 1.0, 1)
            trip_status = random.choices(trip_statuses, weights=[0.8, 0.15, 0.05], k=1)[0]

            request_ts = datetime.now() - timedelta(days=random.randint(1, 90), hours=random.randint(1, 23), minutes=random.randint(1, 59))
            start_ts = request_ts + timedelta(minutes=random.randint(1, 10))
            end_ts = start_ts + timedelta(minutes=random.randint(10, 60))

            is_trip_completed = False
            date_val = request_ts.strftime('%Y-%m-%d')

            if trip_status == 'cancelled':
                end_ts = start_ts
                trip_distance = 0
                trip_fare = 0
                surge_multiplier = 1.0
            elif trip_status == 'in_progress':
                end_ts = None
                trip_fare = 0
            else: # completed
                is_trip_completed = True

            trips_data.append((
                trip_id, user_id, driver_id, vehicle_type, start_loc, end_loc, trip_distance,
                trip_fare, surge_multiplier, trip_status,
                request_ts.strftime('%Y-%m-%d %H:%M:%S'),
                start_ts.strftime('%Y-%m-%d %H:%M:%S'),
                end_ts.strftime('%Y-%m-%d %H:%M:%S') if end_ts else None,
                is_trip_completed,
                date_val
            ))

            if trip_status == 'completed':
                payment_id = f'PMT{i:04d}'
                amount = trip_fare
                method = random.choice(['credit_card', 'cash'])
                payment_ts = end_ts + timedelta(minutes=random.randint(0, 5)) if end_ts else request_ts
                payment_status = 'paid'
                payments_data.append((payment_id, trip_id, amount, method, payment_ts.strftime('%Y-%m-%d %H:%M:%S'), payment_status))

        await cursor.executemany("INSERT INTO uber_trips_data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", trips_data)
        await cursor.executemany("INSERT INTO uber_payments VALUES (?, ?, ?, ?, ?, ?)", payments_data)

        await conn.commit()
    logging.info(f"Database '{DB_NAME}' and tables created and populated with sample data.")

# --- LangGraph State Definition ---
class AgentState(TypedDict):
    user_query: str
    original_user_query: str
    sql_query: Optional[str]
    sql_explanation: Optional[str]
    execution_results: Optional[List[Tuple]]
    execution_description: Optional[List[Tuple]]
    error_message: Optional[str]
    structured_error: Optional[Dict]
    retry_count: int
    detected_workspace: Optional[str]
    selected_tables: Optional[List[str]]
    pruned_columns_map: Optional[Dict[str, List[str]]]
    generated_sql_attempts: List[str]
    has_enough_context: bool

# --- LLM Client (Now Async) ---
class LLMClient:
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        # Use AsyncOpenAI client
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        logging.info(f"Async OpenAI LLM client initialized with model: {self.model}")

    async def chat_completion(self, messages: List[Dict], temperature: float = 0.1, response_format: Optional[Dict] = None) -> Optional[str]:

        formatted_messages = []
        for msg in messages:
            if isinstance(msg, (SystemMessage, HumanMessage, AIMessage)):
                formatted_messages.append({"role": msg.type, "content": msg.content})
            elif isinstance(msg, dict) and "role" in msg and "content" in msg:
                formatted_messages.append(msg)
            else:
                logging.error(f"Invalid message format encountered: {msg}. Skipping.")

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                temperature=temperature,
                response_format=response_format
            )
            return response.choices[0].message.content
        except APIError as e:
            logging.error(f"OpenAI API Error: {e}")
            return None
        except RateLimitError:
            logging.error("OpenAI API Rate Limit Exceeded. Please try again later.")
            return None
        except Exception as e:
            logging.error(f"An unexpected error occurred during LLM call: {e}")
            return None

# --- Base QueryGPT Class with DB and shared methods (Now Async) ---
class BaseQueryGPT:
    def __init__(self, db_name='uber_analytics.db', llm_model: str = "gpt-4o"):
        self.db_name = db_name
        self.llm_client = LLMClient(os.environ.get("OPENAI_API_KEY"), model=llm_model)
        # self.history = [] # History will be managed externally or persistently in production

    async def _execute_query(self, sql_query: str) -> Tuple[Any, Optional[List[Tuple]]]:
        """
        Helper to execute SQL queries and return results and description.
        Returns (error_message_dict, None) or (results_list, description_tuple).
        """
        conn = None
        try:
            conn = await aiosqlite.connect(self.db_name)
            cursor = await conn.cursor()
            await cursor.execute(sql_query)
            results = await cursor.fetchall()
            logging.info(f"Executed SQL: {sql_query}")
            return results, cursor.description
        except sqlite3.OperationalError as e:
            error_msg = f"SQL Operational Error: {e}"
            logging.error(error_msg, exc_info=True)
            return {"error_type": "SQL_OPERATIONAL_ERROR", "message": str(e)}, None
        except sqlite3.Error as e:
            error_msg = f"SQLite Error: {e}"
            logging.error(error_msg, exc_info=True)
            return {"error_type": "SQLITE_ERROR", "message": str(e)}, None
        except Exception as e:
            error_msg = f"Unexpected Error: {type(e).__name__}: {e}"
            logging.error(error_msg, exc_info=True)
            return {"error_type": "GENERAL_EXECUTION_ERROR", "message": str(e)}, None
        finally:
            if conn:
                try:
                    await conn.close()
                except sqlite3.Error as e:
                    logging.error(f"Error closing database connection: {e}")
                except Exception as e:
                    logging.error(f"Unexpected error closing database connection: {e}")
        logging.critical(f"_execute_query reached an unreachable state for query: {sql_query}")
        return {"error_type": "UNREACHABLE_STATE", "message": "Internal error in _execute_query function."}, None

    async def _get_schema_definition_for_llm(self, table_names: List[str], columns_to_exclude: Optional[Dict[str, List[str]]] = None):
        """Generates a detailed schema description for the LLM based on specified tables and columns to exclude."""
        async with aiosqlite.connect(self.db_name) as conn:
            cursor = await conn.cursor()

            schema_info_parts = ["The database `uber_analytics_db` contains the following tables:\n"]

            for table_name in table_names:
                try:
                    await cursor.execute(f"PRAGMA table_info({table_name});")
                    columns_info = await cursor.fetchall()
                except sqlite3.OperationalError:
                    logging.warning(f"Table '{table_name}' not found during schema retrieval. Skipping.")
                    continue

                schema_info_parts.append(f"Table: `{table_name}`\n")
                schema_info_parts.append("Columns:\n")

                cols_to_exclude_for_table = columns_to_exclude.get(table_name, []) if columns_to_exclude else []

                for col in columns_info:
                    col_name = col[1]
                    if col_name in cols_to_exclude_for_table:
                        continue
                    col_type = col[2]
                    pk_info = " (PRIMARY KEY)" if col[5] else ""
                    schema_info_parts.append(f"- `{col_name}` ({col_type}){pk_info}\n")

                if table_name == "uber_trips_data":
                    if "user_id" not in cols_to_exclude_for_table: schema_info_parts.append("- `user_id` is a Foreign Key referencing `uber_users.user_id`.\n")
                    if "driver_id" not in cols_to_exclude_for_table: schema_info_parts.append("- `driver_id` is a Foreign Key referencing `uber_drivers.driver_id`.\n")
                    if ("trip_start_location" not in cols_to_exclude_for_table or "trip_end_location" not in cols_to_exclude_for_table) and "uber_cities" in table_names:
                        schema_info_parts.append("- `trip_start_location` and `trip_end_location` can be linked to `uber_cities.name_of_city`.\n")
                    if "is_trip_completed" not in cols_to_exclude_for_table: schema_info_parts.append("- `is_trip_completed` is a boolean indicating if a trip finished successfully.\n")
                    if "date_val" not in cols_to_exclude_for_table: schema_info_parts.append("- `date_val` is a partition column (DATE type).\n")
                elif table_name == "uber_payments":
                    if "trip_id" not in cols_to_exclude_for_table: schema_info_parts.append("- `trip_id` is a Foreign Key referencing `uber_trips_data.trip_id`.\n")
                elif table_name == "uber_cities":
                    if "name_of_city" not in cols_to_exclude_for_table: schema_info_parts.append("- `name_of_city` is the city's name.\n")

                schema_info_parts.append("\n")

            schema_info_parts.append("Relationships:\n")
            if "uber_trips_data" in table_names and ("uber_users" in table_names or "uber_drivers" in table_names):
                schema_info_parts.append("- `uber_trips_data` table links `uber_users` and `uber_drivers`.\n")
            if "uber_payments" in table_names and "uber_trips_data" in table_names:
                schema_info_parts.append("- `uber_payments` table links to `uber_trips_data`.\n")
            if "uber_trips_data" in table_names and "uber_cities" in table_names:
                schema_info_parts.append("- `uber_trips_data` can be joined with `uber_cities` on location/city names.\n\n")

            schema_info_parts.append("## DATATYPE FUNCTIONS AND COMPARISONS ##\n")
            schema_info_parts.append("It's important to consider the data type of columns. For example date functions like DATE_TRUNC(), DATE_ADD can only be used with TIMESTAMPS or DATEs.\n")
            schema_info_parts.append("SUBSTR() and CONCAT() can only be used on STRINGS.\n\n")

            schema_info_parts.append("Common metrics and notes:\n")
            schema_info_parts.append("- 'Total trips' or 'completed trips' refers to `COUNT(*)` from `uber_trips_data` where `trip_status = 'completed'` or `is_trip_completed = TRUE`.\n")
            schema_info_parts.append("- 'Total revenue' refers to `SUM(trip_fare_usd)` from `uber_trips_data` where `trip_status = 'completed'`.\n")
            schema_info_parts.append("- 'Average trip fare' is `AVG(trip_fare_usd)` from `uber_trips_data` for completed trips.\n")
            schema_info_parts.append("- Dates and timestamps are stored as `YYYY-MM-DD HH:MM:SS` or `YYYY-MM-DD`.\n")
            schema_info_parts.append("- When a user asks for 'list' or 'show', limit results to 10 for brevity.\n")
            schema_info_parts.append("- For numeric queries (e.g., 'fare more than 50'), use the appropriate column (e.g., `trip_fare_usd`).\n")
            schema_info_parts.append("- When filtering by date, use `DATE()` function for date comparison (e.g., `DATE(request_timestamp) = DATE('now')`).\n")
            schema_info_parts.append("- For previous month, use `strftime('%Y-%m-01 00:00:00', DATE('now', '-1 month'))` and `strftime('%Y-%m-01 00:00:00', 'now')`.\n")

        return "".join(schema_info_parts)

    async def _get_table_columns_map(self, table_names: List[str]) -> Dict[str, List[str]]:
        """Helper to get a map of table name to its columns."""
        async with aiosqlite.connect(self.db_name) as conn:
            cursor = await conn.cursor()
            table_cols_map = {}
            for table_name in table_names:
                try:
                    await cursor.execute(f"PRAGMA table_info({table_name});")
                    table_cols_map[table_name] = [col[1] for col in await cursor.fetchall()]
                except sqlite3.OperationalError:
                    logging.warning(f"Table '{table_name}' not found when retrieving column map.")
                    table_cols_map[table_name] = []
            return table_cols_map

    def _get_all_sql_samples(self) -> List[Dict]:
        return [
            {"nl": "Total number of completed trips.", "sql": "SELECT COUNT(*) FROM uber_trips_data WHERE trip_status = 'completed';"},
            {"nl": "What is the average fare for trips from New York?", "sql": "SELECT AVG(trip_fare_usd) FROM uber_trips_data WHERE trip_start_location = 'New York' AND trip_status = 'completed';"},
            {"nl": "List drivers with rating above 4.5 in Los Angeles.", "sql": "SELECT driver_name, rating, city FROM uber_drivers WHERE rating > 4.5 AND city = 'Los Angeles' LIMIT 10;"},
            {"nl": "How many trips were completed by Teslas in Seattle yesterday?", "sql": "SELECT COUNT(*) FROM uber_trips_data WHERE vehicle_type = 'Tesla' AND trip_end_location = 'Seattle' AND trip_status = 'completed' AND DATE(end_timestamp) = DATE('now', '-1 day');"},
            {"nl": "Show me revenue for last month.", "sql": "SELECT SUM(trip_fare_usd) FROM uber_trips_data WHERE trip_status = 'completed' AND request_timestamp >= strftime('%Y-%m-01 00:00:00', DATE('now', '-1 month')) AND request_timestamp < strftime('%Y-%m-01 00:00:00', 'now');"},
            {"nl": "Details of user 'USR001'.", "sql": "SELECT * FROM uber_users WHERE user_id = 'USR001' LIMIT 1;"},
            {"nl": "Trips with surge multiplier greater than 2.0.", "sql": "SELECT trip_id, trip_fare_usd, surge_multiplier FROM uber_trips_data WHERE surge_multiplier > 2.0 AND trip_status = 'completed' LIMIT 10;"},
            {"nl": "Find the highest rated driver.", "sql": "SELECT driver_name, rating FROM uber_drivers ORDER BY rating DESC LIMIT 1;"},
            {"nl": "Number of users who signed up in Chicago.", "sql": "SELECT COUNT(DISTINCT T1.user_id) FROM uber_users AS T1 JOIN uber_trips_data AS T2 ON T1.user_id = T2.user_id WHERE T2.trip_start_location = 'Chicago';"},
            {"nl": "Total amount paid via credit card.", "sql": "SELECT SUM(amount_usd) FROM uber_payments WHERE payment_method = 'credit_card' AND status = 'paid';"},
            {"nl": "How many trips are currently in progress?", "sql": "SELECT COUNT(*) FROM uber_trips_data WHERE trip_status = 'in_progress';"},
            {"nl": "List all cities where Uber operates.", "sql": "SELECT name_of_city FROM uber_cities WHERE is_operational = TRUE LIMIT 10;"}
        ]

# --- QueryGPT V5 with LangGraph (Self-Correction) ---
class QueryGPT_V5_LangGraph(BaseQueryGPT):
    MAX_RETRIES = 3
    NUM_RAG_SAMPLES = 5

    def __init__(self, db_name='uber_analytics.db', llm_model: str = "gpt-4o"):
        super().__init__(db_name, llm_model)
        self.system_workspaces = self._define_system_workspaces()
        self.all_sql_samples = self._get_all_sql_samples()
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()
        logging.info("QueryGPT V5 (LangGraph) initialized with self-correction and enhanced RAG.")

    def _define_system_workspaces(self):
        return {
            "mobility": {
                "description": "Queries related to trips, drivers, users, vehicles, payments, and city data.",
                "tables": ["uber_trips_data", "uber_drivers", "uber_users", "uber_payments", "uber_cities"],
            },
            "finance": {
                "description": "Queries related to revenue, payments, and financial reconciliation.",
                "tables": ["uber_payments", "uber_trips_data"],
            }
        }

    # --- LangGraph Nodes (Agentic Functions) ---
    async def _call_llm_agent(self, agent_name: str, messages: List[Dict], temperature: float = 0.1, response_format: Optional[Dict] = None) -> Dict:
        """Helper to call the LLM for a specific agent and parse its JSON response."""
        logging.info(f"Calling LLM for {agent_name}...")

        llm_response_content = await self.llm_client.chat_completion(
            messages=messages,
            temperature=temperature,
            response_format=response_format
        )
        if llm_response_content:
            try:
                return json.loads(llm_response_content)
            except json.JSONDecodeError as e:
                logging.error(f"{agent_name} - Failed to parse LLM response JSON: {e}. Raw content: {llm_response_content[:200]}...")
                return {"error": f"LLM response not in expected JSON format: {llm_response_content[:100]}..."}
        return {"error": f"{agent_name} - LLM did not return a response."}

    async def _initial_input_validation(self, state: AgentState) -> Dict:
        logging.info("Node: Initial Input Validation...")
        user_query = state['user_query'].strip()

        if not user_query or len(user_query) < 5:
            logging.warning(f"User query '{user_query}' too short or empty.")
            return {"error_message": "User query is too short or empty. Please provide more details.", "has_enough_context": False}

        return {"user_query": user_query, "has_enough_context": True}


    async def _determine_intent(self, state: AgentState) -> Dict:
        logging.info("Node: Determining Intent...")
        user_query = state['user_query']

        intent_system_prompt = """
You are an Intent Agent. Your task is to identify the most relevant business domain/workspace for a given user query.
Choose from 'mobility' or 'finance'.
If the query does not clearly fit into either, default to 'mobility'.
Respond ONLY with a JSON object containing the 'workspace' key and a 'confidence' score (0.0-1.0).
Example: {"workspace": "mobility", "confidence": 0.9}
"""
        intent_response = await self._call_llm_agent(
            "IntentAgent",
            messages=[
                {"role": "system", "content": intent_system_prompt},
                {"role": "user", "content": user_query}
            ],
            response_format={"type": "json_object"}
        )

        detected_workspace = intent_response.get("workspace", "mobility")
        if "error" in intent_response:
            logging.error(f"Intent Agent error: {intent_response['error']}. Falling back to 'mobility'.")
            detected_workspace = "mobility"

        return {"detected_workspace": detected_workspace}

    async def _select_tables(self, state: AgentState) -> Dict:
        logging.info("Node: Selecting Tables...")
        user_query = state['user_query']
        detected_workspace = state['detected_workspace']

        relevant_tables_for_workspace = self.system_workspaces.get(detected_workspace, {}).get("tables", [])

        table_agent_system_prompt = f"""
You are a Table Agent. Given a user query and a list of candidate tables from the detected workspace,
identify the most relevant tables needed to answer the query.
Available tables in the '{detected_workspace}' workspace: {', '.join(relevant_tables_for_workspace)}.
Prioritize tables directly mentioned or strongly implied by the query.
If no specific tables are strongly implied, select a reasonable minimal set that might contain relevant data.
Respond ONLY with a JSON object containing a 'tables' array (list of table names) and an 'explanation'.
Example: {{"tables": ["uber_trips_data", "uber_drivers"], "explanation": "Query involves trips and driver information."}}
"""
        table_agent_response = await self._call_llm_agent(
            "TableAgent",
            messages=[
                 {"role": "system", "content": table_agent_system_prompt},
                 {"role": "user", "content": user_query}
            ],
            response_format={"type": "json_object"}
        )

        selected_tables_llm = table_agent_response.get("tables", [])
        if "error" in table_agent_response:
            logging.error(f"Table Agent error: {table_agent_response['error']}. Attempting fallback.")
            selected_tables_llm = []

        final_selected_tables = [t for t in selected_tables_llm if t in relevant_tables_for_workspace]
        if not final_selected_tables:
            final_selected_tables = relevant_tables_for_workspace
            logging.warning("Table Agent failed to select specific tables or selected irrelevant ones. Falling back to all tables in workspace.")
            if not final_selected_tables:
                logging.error(f"No tables found for workspace '{detected_workspace}'.")
                return {"error_message": f"No tables found for workspace '{detected_workspace}' or general context.", "has_enough_context": False}

        return {"selected_tables": final_selected_tables, "has_enough_context": True}

    async def _prune_columns(self, state: AgentState) -> Dict:
        logging.info("Node: Pruning Columns...")
        user_query = state['user_query']
        selected_tables = state['selected_tables']

        full_table_columns_map = await self._get_table_columns_map(selected_tables)

        column_prune_system_prompt = f"""
You are a Column Prune Agent. Given a user query and the full schemas of relevant tables,
identify and list ONLY the columns that are NOT directly relevant to answering the query for each table.
The goal is to reduce the schema size for the main SQL generation LLM, while retaining essential columns for joins or general context.

**Important Considerations:**
- Always keep primary key columns (e.g., `_id` columns) and foreign key columns.
- Keep columns that are directly referenced or strongly implied by the user's query.
- Keep columns that are commonly used for filtering or grouping, even if not explicitly mentioned (e.g., `trip_status`, `date_val`).
- Only remove columns that are highly unlikely to be needed for the query.

Schemas: {json.dumps(full_table_columns_map)}

Respond ONLY with a JSON object containing a 'pruned_schema' dictionary, where keys are table names and values are arrays of column names to *exclude*. Also provide an 'explanation'.
Example: {{"pruned_schema": {{"uber_trips_data": ["user_id", "driver_id"]}}, "explanation": "Removed irrelevant IDs."}}
"""
        column_prune_response = await self._call_llm_agent(
            "ColumnPruneAgent",
            messages=[
                 {"role": "system", "content": column_prune_system_prompt},
                 {"role": "user", "content": user_query}
            ],
            response_format={"type": "json_object"}
        )

        columns_to_exclude_from_llm = column_prune_response.get("pruned_schema", {})
        if "error" in column_prune_response:
            logging.error(f"Column Prune Agent error: {column_prune_response['error']}. Skipping pruning.")
            columns_to_exclude_from_llm = {}

        return {"pruned_columns_map": columns_to_exclude_from_llm}

    async def _generate_sql(self, state: AgentState) -> Dict:
        logging.info("Node: Generating SQL...")
        user_query = state['user_query']
        selected_tables = state['selected_tables']
        pruned_columns_map = state['pruned_columns_map']
        detected_workspace = state['detected_workspace']

        schema_definition_for_llm = await self._get_schema_definition_for_llm(selected_tables, pruned_columns_map)

        # Enhanced RAG: Dynamically select a diverse set of samples
        available_samples = self.all_sql_samples
        selected_rag_samples = random.sample(available_samples, min(self.NUM_RAG_SAMPLES, len(available_samples)))
        relevant_samples_str = "\n".join([f"User: {s['nl']}\nSQL: {s['sql']}" for s in selected_rag_samples])

        rectification_context = ""
        if state['error_message']:
            rectification_context += f"\n-- Previous SQL Attempt:\n{state['sql_query']}\n"
            rectification_context += f"\n-- Error encountered during last execution:\nError Type: {state['structured_error'].get('error_type', 'UNKNOWN')}\nError Message: {state['structured_error'].get('message', 'No specific message provided.')}\n"
            rectification_context += "\nPlease correct the SQL query based on this feedback. Do NOT repeat the same mistake. Focus on resolving the mentioned error.\n"
            logging.info(f"Rectification attempt {state['retry_count'] + 1} for query: {user_query}")

        query_gen_system_prompt = f"""
You are an expert SQLite SQL generator for an Uber data analytics database.
Your task is to convert natural language questions into executable SQLite SQL queries.

**Database Schema (relevant and pruned):**
{schema_definition_for_llm}

**Examples of Natural Language to SQL mappings (Few-Shot Prompting):**
{relevant_samples_str}

**Instructions for query generation:**
1.  Generate only the SQL query.
2.  Assume `trip_status = 'completed'` or `is_trip_completed = TRUE` for all aggregations (SUM, AVG, COUNT) on trips or fares, unless the user explicitly asks for 'cancelled', 'in progress', or 'all' trips.
3.  For 'list' or 'show' queries, add `LIMIT 10;` to the end of the query for brevity.
4.  Be precise with column names and table names as provided in the schema.
5.  Handle date/time filters appropriately using SQLite date functions (e.g., `DATE('now', '-1 day')`, `strftime`).
6.  If the query involves multiple tables, use appropriate JOINs.
7.  If you are rectifying a previous error, carefully analyze the error type and message and the previous SQL to make the necessary corrections. Do NOT repeat the same mistake.
8.  Respond ONLY with a JSON object containing the `sql_query` and `explanation` keys.
    Example: {{"sql_query": "SELECT COUNT(*) FROM uber_trips_data;", "explanation": "Counts all trips."}}

{rectification_context}
"""
        messages = [
            {"role": "system", "content": query_gen_system_prompt},
            {"role": "user", "content": user_query}
        ]

        query_gen_response = await self._call_llm_agent(
            "QueryGenerationAgent",
            messages=messages,
            response_format={"type": "json_object"}
        )

        generated_sql = query_gen_response.get("sql_query")
        explanation_text = query_gen_response.get("explanation")

        if "error" in query_gen_response or not generated_sql:
            logging.error(f"SQL Generation Agent error: {query_gen_response.get('error', 'Unknown')}")
            return {"error_message": f"SQL Generation failed: {query_gen_response.get('error', 'Unknown')}", "has_enough_context": False}

        if not generated_sql.strip().endswith(';'): generated_sql += ';'

        generated_sql_attempts = state.get('generated_sql_attempts', [])
        generated_sql_attempts.append(generated_sql)

        return {
            "sql_query": generated_sql,
            "sql_explanation": explanation_text,
            "generated_sql_attempts": generated_sql_attempts,
            "error_message": None,
            "structured_error": None,
            "has_enough_context": True
        }

    async def _validate_sql_syntax(self, state: AgentState) -> Dict:
        logging.info("Node: Validating SQL Syntax...")
        sql_query = state['sql_query']

        if not sql_query:
            return {"error_message": "No SQL query generated for validation.", "structured_error": {"error_type": "NO_SQL_GENERATED", "message": "SQL query was empty or None."}, "has_enough_context": True}

        try:
            parse_one(sql_query, dialect="sqlite")
            logging.info("SQL syntax is valid.")
            return {"error_message": None, "structured_error": None, "has_enough_context": True}
        except ParseError as e:
            error_msg = f"SQL Syntax Error: {e}"
            logging.warning(error_msg)
            return {"error_message": "SQL Syntax Error", "structured_error": {"error_type": "SQL_SYNTAX_ERROR", "message": str(e)}, "has_enough_context": True}
        except Exception as e:
            error_msg = f"Unexpected error during SQL syntax validation: {type(e).__name__}: {e}"
            logging.error(error_msg, exc_info=True)
            return {"error_message": "Unexpected Syntax Validation Error", "structured_error": {"error_type": "SYNTAX_VALIDATION_ERROR", "message": str(e)}, "has_enough_context": True}


    async def _execute_sql(self, state: AgentState) -> Dict:
        logging.info("Node: Executing SQL...")
        sql_query = state['sql_query']

        results, db_error_or_description = await self._execute_query(sql_query)

        if isinstance(results, dict) and "error_type" in results:
            logging.warning(f"SQL Execution failed with structured error: {results}")
            return {"error_message": results["message"], "structured_error": results, "has_enough_context": True}
        else:
            logging.info("SQL Execution successful.")
            return {
                "execution_results": results,
                "execution_description": db_error_or_description,
                "error_message": None,
                "structured_error": None,
                "has_enough_context": True
            }

    # --- LangGraph Conditional Edges ---
    def _should_retry(self, state: AgentState) -> str:
        logging.info(f"Node: Checking if retry needed. Retry count: {state['retry_count']}, Error: {state['error_message']}")

        if state['error_message'] and state['retry_count'] < self.MAX_RETRIES:
            state['retry_count'] += 1
            logging.info(f"Error detected and retries remaining. Incrementing retry_count to {state['retry_count']}. Retrying SQL generation.")
            return "retry"
        elif state['error_message'] and state['retry_count'] >= self.MAX_RETRIES:
            logging.info("Error detected but max retries reached. Ending with error.")
            return "end_with_error"
        else:
            logging.info("No error, or error resolved. Ending successfully.")
            return "end_successfully"

    def _should_continue_workflow(self, state: AgentState) -> str:
        """
        Determines if the workflow should continue based on initial validation or context issues.
        """
        if not state['has_enough_context'] and state['error_message']:
            return "end_with_message"
        return "continue"

    # --- LangGraph Workflow Definition ---
    def _build_workflow(self):
        workflow = StateGraph(AgentState)

        # Define the nodes (all are async now)
        workflow.add_node("initial_input_validation", self._initial_input_validation)
        workflow.add_node("determine_intent", self._determine_intent)
        workflow.add_node("select_tables", self._select_tables)
        workflow.add_node("prune_columns", self._prune_columns)
        workflow.add_node("generate_sql", self._generate_sql)
        workflow.add_node("validate_sql_syntax", self._validate_sql_syntax)
        workflow.add_node("execute_sql", self._execute_sql)

        # Define the edges
        workflow.set_entry_point("initial_input_validation")

        workflow.add_conditional_edges(
            "initial_input_validation",
            self._should_continue_workflow,
            {
                "end_with_message": END,
                "continue": "determine_intent"
            }
        )

        workflow.add_edge("determine_intent", "select_tables")

        workflow.add_conditional_edges(
            "select_tables",
            self._should_continue_workflow,
            {
                "end_with_message": END,
                "continue": "prune_columns"
            }
        )

        workflow.add_edge("prune_columns", "generate_sql")

        workflow.add_edge("generate_sql", "validate_sql_syntax")

        workflow.add_conditional_edges(
            "validate_sql_syntax",
            self._should_retry,
            {
                "retry": "generate_sql",
                "end_with_error": END,
                "end_successfully": "execute_sql"
            }
        )

        workflow.add_conditional_edges(
            "execute_sql",
            self._should_retry,
            {
                "retry": "generate_sql",
                "end_with_error": END,
                "end_successfully": END
            }
        )

        return workflow

    async def query(self, natural_language_query: str, explain_sql: bool = False, debug_mode: bool = False) -> Dict:
        logging.info(f"Starting LangGraph workflow for query: '{natural_language_query}'")

        initial_state = AgentState(
            user_query=natural_language_query,
            original_user_query=natural_language_query,
            sql_query=None,
            sql_explanation=None,
            execution_results=None,
            execution_description=None,
            error_message=None,
            structured_error=None,
            retry_count=0,
            detected_workspace=None,
            selected_tables=None,
            pruned_columns_map=None,
            generated_sql_attempts=[],
            has_enough_context=True
        )

        final_state: Optional[AgentState] = None
        try:
            final_state = await self.app.ainvoke(initial_state, config={"callbacks": [OpenTelemetryCallbackHandler()], "debug": debug_mode})
        except Exception as e:
            logging.error(f"LangGraph workflow execution failed: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Internal workflow error: {type(e).__name__}: {e}",
                "sql": None,
                "attempts": [],
                "results": None,
                "columns": None
            }

        if not isinstance(final_state, dict):
            logging.error(f"LangGraph returned unexpected non-dict final_state: {final_state}")
            return {
                "success": False,
                "message": "Unexpected state returned from workflow.",
                "sql": None,
                "attempts": [],
                "results": None,
                "columns": None
            }

        final_sql = final_state.get('sql_query')
        all_sql_attempts = final_state.get('generated_sql_attempts', [])
        final_results = final_state.get('execution_results')
        final_description = final_state.get('execution_description')
        final_error = final_state.get('error_message')
        retries_taken = final_state.get('retry_count', 0)
        sql_explanation = final_state.get('sql_explanation')

        response_data: Dict[str, Any] = {
            "query": natural_language_query,
            "sql": final_sql,
            "all_sql_attempts": all_sql_attempts,
            "retries_taken": retries_taken,
            "explanation": sql_explanation if explain_sql else None,
            "debug_info": final_state if debug_mode else None # Include full state for debug
        }

        if not final_state.get('has_enough_context', True) or final_error:
            response_data.update({
                "success": False,
                "message": final_error if final_error else "QueryGPT did not have enough context or ability to resolve this query.",
                "results": None,
                "columns": None,
                "error_details": final_state.get('structured_error')
            })
        else:
            columns = [col[0] for col in final_description] if final_description else []
            response_data.update({
                "success": True,
                "message": "Query executed successfully.",
                "results": final_results,
                "columns": columns
            })

        return response_data

# Removed the interactive CLI demonstration from here.
# It will be handled by the FastAPI application.
# if __name__ == "__main__":
#    asyncio.run(main())
