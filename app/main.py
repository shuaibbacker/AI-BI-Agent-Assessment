"""FastAPI Backend - API endpoints for the BI Agent."""

import os
import logging
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.agent import BIAgent
from app.tools import get_tool_registry, ToolRegistry
from app.monday_client import get_monday_client, MondayAPIError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default"


class QueryResponse(BaseModel):
    response: str
    tool_trace: List[Dict]
    iterations: int
    error: Optional[str] = None


class BoardInfo(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    state: Optional[str] = None


class AnalysisRequest(BaseModel):
    board_id: int
    analysis_type: str = "full"


class AnalysisResponse(BaseModel):
    board_id: int
    board_name: str
    results: Dict
    tool_trace: List[Dict]


# Global agent instance (per session could be implemented)
agent_instances: Dict[str, BIAgent] = {}


def get_or_create_agent(session_id: str) -> BIAgent:
    """Get or create an agent instance for a session."""
    if session_id not in agent_instances:
        agent_instances[session_id] = BIAgent()
        logger.info(f"Created new agent for session {session_id}")
    return agent_instances[session_id]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    logger.info("BI Agent API starting up...")
    yield
    logger.info("BI Agent API shutting down...")


# Create FastAPI app
app = FastAPI(
    title="BI Agent API",
    description="Business Intelligence Agent with Monday.com integration",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "BI Agent API",
        "version": "1.0.0",
        "endpoints": [
            "/query",
            "/boards",
            "/analyze",
            "/health"
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Test Monday.com connection
        client = get_monday_client()
        boards = client.get_boards(limit=1)
        monday_status = "connected"
    except Exception as e:
        monday_status = f"error: {str(e)}"
    
    return {
        "status": "healthy",
        "monday_api": monday_status,
        "active_sessions": len(agent_instances)
    }


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a natural language query through the BI Agent."""
    try:
        logger.info(f"Processing query for session {request.session_id}: {request.query}")
        
        agent = get_or_create_agent(request.session_id)
        result = agent.process_query(request.query)
        
        return QueryResponse(
            response=result["response"],
            tool_trace=result["tool_trace"],
            iterations=result["iterations"],
            error=result.get("error")
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/boards")
async def list_boards(limit: int = 10):
    """List available Monday.com boards."""
    try:
        client = get_monday_client()
        boards = client.get_boards(limit=limit)
        return {"boards": boards, "count": len(boards)}
    except MondayAPIError as e:
        logger.error(f"Monday API error: {str(e)}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Error listing boards: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/boards/{board_id}")
async def get_board(board_id: int):
    """Get details for a specific board."""
    try:
        client = get_monday_client()
        
        # Get board data
        board_data = client.get_all_board_items(board_id)
        columns = client.get_board_columns(board_id)
        
        return {
            "board_id": board_id,
            "board_name": board_data["board_name"],
            "total_items": board_data["total_count"],
            "columns": [{"id": c["id"], "title": c["title"], "type": c["type"]} for c in columns],
            "sample_items": board_data["items"][:5] if board_data["items"] else []
        }
        
    except MondayAPIError as e:
        logger.error(f"Monday API error: {str(e)}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting board: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze")
async def analyze_board(request: AnalysisRequest):
    """Run analysis on a specific board."""
    try:
        tools = get_tool_registry()
        
        # First fetch the board
        fetch_result = tools.execute_tool("fetch_board_data", {"board_id": request.board_id})
        
        if "error" in fetch_result:
            raise HTTPException(status_code=400, detail=fetch_result["error"])
        
        # Run requested analysis
        if request.analysis_type == "pipeline":
            result = tools.execute_tool("analyze_pipeline", {"board_id": request.board_id})
        elif request.analysis_type == "conversions":
            result = tools.execute_tool("analyze_conversions", {"board_id": request.board_id})
        elif request.analysis_type == "sector":
            result = tools.execute_tool("analyze_sector_breakdown", {"board_id": request.board_id})
        elif request.analysis_type == "quarterly":
            result = tools.execute_tool("analyze_quarterly_trends", {"board_id": request.board_id})
        elif request.analysis_type == "forecast":
            result = tools.execute_tool("compute_forecast", {"board_id": request.board_id})
        elif request.analysis_type == "closed":
            result = tools.execute_tool("get_closed_revenue", {"board_id": request.board_id})
        else:
            # Full analysis
            result = {
                "board_id": request.board_id,
                "board_name": fetch_result["board_name"],
                "analyses": {}
            }
            from app.analytics import analyze_boards
            for analysis_type in ["pipeline", "conversions", "sector", "quarterly", "forecast", "closed"]:
                try:
                    analysis_result = tools.execute_tool(f"analyze_{analysis_type}", {"board_id": request.board_id})
                    result["analyses"][analysis_type] = analysis_result
                except:
                    pass
        
        return {
            "board_id": request.board_id,
            "board_name": fetch_result.get("board_name", "Unknown"),
            "results": result,
            "tool_trace": tools.get_tool_traces()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing board: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/cross-board")
async def analyze_multiple_boards(board_ids: List[int]):
    """Analyze multiple boards together."""
    try:
        tools = get_tool_registry()
        result = tools.execute_tool("cross_board_analysis", {"board_ids": board_ids})
        
        return {
            "board_ids": board_ids,
            "results": result,
            "tool_trace": tools.get_tool_traces()
        }
        
    except Exception as e:
        logger.error(f"Error in cross-board analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}/trace")
async def get_session_trace(session_id: str):
    """Get tool trace for a session."""
    if session_id not in agent_instances:
        return {"error": "Session not found"}
    
    agent = agent_instances[session_id]
    return {
        "session_id": session_id,
        "conversation_summary": agent.get_conversation_summary(),
        "recent_trace": agent.current_trace[-10:] if agent.current_trace else []
    }


@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear a session and its history."""
    if session_id in agent_instances:
        del agent_instances[session_id]
        return {"message": f"Session {session_id} cleared"}
    return {"message": "Session not found"}


@app.get("/tools")
async def list_tools():
    """List available tools."""
    tools = get_tool_registry()
    definitions = tools.get_tool_definitions()
    
    return {
        "tools": [
            {
                "name": tool["function"]["name"],
                "description": tool["function"]["description"]
            }
            for tool in definitions
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
