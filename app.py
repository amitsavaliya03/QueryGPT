# app.py

from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
import os
import logging

# Import the QueryGPT agent and database setup function
from querygpt_agent import QueryGPT_V5_LangGraph, setup_database_if_not_exists, DB_NAME

# Configure basic logging for FastAPI, will be merged with QueryGPT's structured logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app instance
app = FastAPI(
    title="QueryGPT API",
    description="An API for converting natural language to SQL and executing it against the Uber analytics database.",
    version="1.0.0",
)

# Global instance of QueryGPT (initialized on startup)
query_gpt_instance: Optional[QueryGPT_V5_LangGraph] = None

# Pydantic models for request and response
class QueryRequest(BaseModel):
    query: str = Field(..., example="Show me total completed trips last month.")
    explain_sql: bool = Field(False, description="Whether to include an explanation of the generated SQL in the response.")
    debug_mode: bool = Field(False, description="Whether to include full internal debug information (e.g., LangGraph state) in the response. **For development only.**")

class QuerySuccessResponse(BaseModel):
    success: bool = True
    message: str = Field(..., example="Query executed successfully.")
    query: str = Field(..., example="Show me total completed trips last month.")
    sql: Optional[str] = Field(None, example="SELECT COUNT(*) FROM uber_trips_data WHERE trip_status = 'completed';")
    explanation: Optional[str] = Field(None, example="Counts the number of completed trips.")
    results: List[List[Any]] = Field(..., example=[[150]])
    columns: List[str] = Field(..., example=["COUNT(*)"])
    all_sql_attempts: List[str] = Field([], example=["SELECT * FROM trips;", "SELECT COUNT(*) FROM uber_trips_data;"])
    retries_taken: int = Field(0, example=1)
    debug_info: Optional[Dict[str, Any]] = None # Full LangGraph state for debugging

class QueryErrorResponse(BaseModel):
    success: bool = False
    message: str = Field(..., example="User query is too short or empty. Please provide more details.")
    query: str = Field(..., example="abc")
    sql: Optional[str] = Field(None, example="SELECT * FRM uber_drivers;")
    all_sql_attempts: List[str] = Field([], example=["SELECT * FRM uber_drivers;"])
    retries_taken: int = Field(0, example=1)
    error_details: Optional[Dict[str, Any]] = None # Structured error from QueryGPT agent
    debug_info: Optional[Dict[str, Any]] = None # Full LangGraph state for debugging

# FastAPI startup event
@app.on_event("startup")
async def startup_event():
    """
    On application startup, set up the database and initialize the QueryGPT agent.
    """
    global query_gpt_instance
    logger.info("Application startup: Setting up database and initializing QueryGPT agent...")

    # Ensure OpenAI API key is set
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        logger.critical("OPENAI_API_KEY environment variable not set. QueryGPT will not function.")
        raise RuntimeError("OPENAI_API_KEY environment variable not set.")

    # Setup database
    try:
        await setup_database_if_not_exists()
        logger.info(f"Database {DB_NAME} ready.")
    except Exception as e:
        logger.critical(f"Failed to set up database: {e}", exc_info=True)
        raise RuntimeError(f"Database setup failed: {e}")

    # Initialize QueryGPT agent
    try:
        query_gpt_instance = QueryGPT_V5_LangGraph(db_name=DB_NAME)
        logger.info("QueryGPT agent initialized successfully.")
    except Exception as e:
        logger.critical(f"Failed to initialize QueryGPT agent: {e}", exc_info=True)
        raise RuntimeError(f"QueryGPT agent initialization failed: {e}")

# FastAPI shutdown event (optional for now, more relevant with connection pools)
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutdown.")

@app.post(
    "/query",
    response_model=QuerySuccessResponse, # Default success response
    responses={
        status.HTTP_200_OK: {"model": QuerySuccessResponse, "description": "Query processed successfully."},
        status.HTTP_400_BAD_REQUEST: {"model": QueryErrorResponse, "description": "Client-side error (e.g., malformed query, insufficient context)."},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": QueryErrorResponse, "description": "Server-side error during query processing or internal workflow."},
    },
    summary="Process a natural language query to generate and execute SQL."
)
async def process_query(request: QueryRequest):
    """
    Processes a natural language query by converting it to SQL, executing it,
    and returning the results. Includes self-correction mechanisms.
    """
    if query_gpt_instance is None:
        logger.error("QueryGPT instance not initialized. This should not happen after startup.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Service not ready: QueryGPT instance not initialized."
        )

    try:
        response_data = await query_gpt_instance.query(
            natural_language_query=request.query,
            explain_sql=request.explain_sql,
            debug_mode=request.debug_mode
        )

        if response_data["success"]:
            # Successfully processed query
            return QuerySuccessResponse(**response_data)
        else:
            # QueryGPT reported an error, map it to a suitable HTTP status
            # For simplicity, if QueryGPT returns 'success: False', it's often a client-side (400) type issue or an unrecoverable internal error.
            # We'll use 400 for general QueryGPT-reported errors that indicate an inability to fulfill the request.
            # For deeper internal workflow errors that prevent processing entirely, a 500 would be more appropriate.
            # Here, the `query` method already catches critical internal errors and reports them with success=False,
            # so we'll treat most 'success: False' as 400 unless specifically categorized as an internal server error.
            http_status = status.HTTP_400_BAD_REQUEST
            if "Internal workflow error" in response_data["message"]:
                 http_status = status.HTTP_500_INTERNAL_SERVER_ERROR

            raise HTTPException(
                status_code=http_status,
                detail=QueryErrorResponse(**response_data).dict() # Use .dict() to convert Pydantic model to dict for detail
            )

    except HTTPException as e:
        # Re-raise explicit HTTPExceptions (e.g., if QueryGPT instance is None)
        raise e
    except Exception as e:
        # Catch any unexpected exceptions during API processing
        logger.error(f"Unexpected error processing query '{request.query}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "message": f"An unexpected server error occurred: {type(e).__name__}: {e}",
                "query": request.query,
                "sql": None,
                "all_sql_attempts": [],
                "retries_taken": 0,
                "error_details": {"type": "UNEXPECTED_API_ERROR", "message": str(e)},
                "debug_info": None # No debug info for unexpected API errors unless explicitly added
            }
        )

# Health check endpoint
@app.get("/health")
async def health_check():
    if query_gpt_instance is not None:
        return {"status": "ok", "message": "QueryGPT API is running and initialized."}
    return {"status": "initializing", "message": "QueryGPT API is starting up."}
