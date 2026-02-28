"""Tools Registry - Defines all tools the agent can call to fetch and analyze data."""

import json
import logging
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime

from app.monday_client import MondayClient, get_monday_client
from app.data_clean import DataCleaner, clean_board_data
from app.analytics import BIAnalytics

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry of all tools available to the BI Agent."""
    
    def __init__(self):
        self.tools: Dict[str, Dict] = {}
        self.handlers: Dict[str, Callable] = {}
        self.tool_traces: List[Dict] = []
        self._register_all_tools()
    
    def _log_tool_call(self, tool_name: str, params: Dict, result: Any):
        """Log tool call for traceability."""
        trace = {
            "timestamp": datetime.now().isoformat(),
            "tool": tool_name,
            "parameters": params,
            "result_summary": self._summarize_result(result)
        }
        self.tool_traces.append(trace)
        logger.info(f"Tool called: {tool_name} with params {params}")
    
    def _summarize_result(self, result: Any) -> Dict:
        """Create a summary of tool result for logging."""
        if isinstance(result, dict):
            return {
                "keys": list(result.keys()),
                "has_error": "error" in result,
                "size": len(str(result))
            }
        return {"type": type(result).__name__}
    
    def get_tool_definitions(self) -> List[Dict]:
        """Get OpenAI function calling tool definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "list_boards",
                    "description": "List all available Monday.com boards the user has access to",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of boards to return",
                                "default": 10
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "fetch_board_data",
                    "description": "Fetch all items from a specific Monday.com board with pagination. MUST call this before any analysis.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "board_id": {
                                "type": "integer",
                                "description": "The numeric ID of the Monday.com board to fetch"
                            }
                        },
                        "required": ["board_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "fetch_multiple_boards",
                    "description": "Fetch data from multiple boards for cross-board analysis",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "board_ids": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "description": "List of board IDs to fetch"
                            }
                        },
                        "required": ["board_ids"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_pipeline",
                    "description": "Analyze sales pipeline metrics including total value, stage breakdown, and average deal size",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "board_id": {
                                "type": "integer",
                                "description": "Board ID to analyze (must be fetched first)"
                            },
                            "include_closed": {
                                "type": "boolean",
                                "description": "Whether to include closed deals",
                                "default": False
                            }
                        },
                        "required": ["board_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_conversions",
                    "description": "Compute conversion rates between pipeline stages",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "board_id": {
                                "type": "integer",
                                "description": "Board ID to analyze"
                            }
                        },
                        "required": ["board_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_sector_breakdown",
                    "description": "Analyze metrics broken down by sector/industry",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "board_id": {
                                "type": "integer",
                                "description": "Board ID to analyze"
                            }
                        },
                        "required": ["board_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_quarterly_trends",
                    "description": "Analyze quarterly trends and QoQ growth",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "board_id": {
                                "type": "integer",
                                "description": "Board ID to analyze"
                            }
                        },
                        "required": ["board_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "compute_forecast",
                    "description": "Compute weighted revenue forecast based on stage probabilities",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "board_id": {
                                "type": "integer",
                                "description": "Board ID to analyze"
                            }
                        },
                        "required": ["board_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_closed_revenue",
                    "description": "Get closed won and lost revenue with win rates",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "board_id": {
                                "type": "integer",
                                "description": "Board ID to analyze"
                            },
                            "time_period": {
                                "type": "string",
                                "description": "Time period filter: this_month, last_month, this_quarter, last_quarter, ytd, last_30_days, last_90_days",
                                "enum": ["this_month", "last_month", "this_quarter", "last_quarter", "ytd", "last_30_days", "last_90_days"]
                            }
                        },
                        "required": ["board_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "cross_board_analysis",
                    "description": "Analyze and compare multiple boards together",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "board_ids": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "description": "List of board IDs to analyze together"
                            }
                        },
                        "required": ["board_ids"]
                    }
                }
            }
        ]
    
    def _register_all_tools(self):
        """Register all tool handlers."""
        self.handlers["list_boards"] = self._handle_list_boards
        self.handlers["fetch_board_data"] = self._handle_fetch_board_data
        self.handlers["fetch_multiple_boards"] = self._handle_fetch_multiple_boards
        self.handlers["analyze_pipeline"] = self._handle_analyze_pipeline
        self.handlers["analyze_conversions"] = self._handle_analyze_conversions
        self.handlers["analyze_sector_breakdown"] = self._handle_analyze_sector_breakdown
        self.handlers["analyze_quarterly_trends"] = self._handle_analyze_quarterly_trends
        self.handlers["compute_forecast"] = self._handle_compute_forecast
        self.handlers["get_closed_revenue"] = self._handle_get_closed_revenue
        self.handlers["cross_board_analysis"] = self._handle_cross_board_analysis
    
    def execute_tool(self, tool_name: str, params: Dict) -> Dict:
        """Execute a tool by name with given parameters."""
        if tool_name not in self.handlers:
            return {"error": f"Unknown tool: {tool_name}"}
        
        try:
            result = self.handlers[tool_name](params)
            self._log_tool_call(tool_name, params, result)
            return result
        except Exception as e:
            error_result = {"error": str(e)}
            self._log_tool_call(tool_name, params, error_result)
            return error_result
    
    def _handle_list_boards(self, params: Dict) -> Dict:
        """List available boards."""
        client = get_monday_client()
        limit = params.get("limit", 10)
        boards = client.get_boards(limit=limit)
        return {
            "boards": boards,
            "count": len(boards)
        }
    
    def _handle_fetch_board_data(self, params: Dict) -> Dict:
        """Fetch data from a single board."""
        board_id = params["board_id"]
        client = get_monday_client()
        
        board_data = client.get_all_board_items(board_id)
        columns = client.get_board_columns(board_id)
        board_data["columns"] = columns
        
        # Store for later analysis
        self._last_fetched_data = {board_id: board_data}
        
        return {
            "board_id": board_id,
            "board_name": board_data["board_name"],
            "total_items": board_data["total_count"],
            "columns": [{"id": c["id"], "title": c["title"], "type": c["type"]} for c in columns],
            "sample_items": board_data["items"][:3] if board_data["items"] else []
        }
    
    def _handle_fetch_multiple_boards(self, params: Dict) -> Dict:
        """Fetch data from multiple boards."""
        board_ids = params["board_ids"]
        client = get_monday_client()
        
        results = []
        self._last_fetched_data = {}
        
        for board_id in board_ids:
            try:
                board_data = client.get_all_board_items(board_id)
                columns = client.get_board_columns(board_id)
                board_data["columns"] = columns
                self._last_fetched_data[board_id] = board_data
                
                results.append({
                    "board_id": board_id,
                    "board_name": board_data["board_name"],
                    "total_items": board_data["total_count"]
                })
            except Exception as e:
                results.append({
                    "board_id": board_id,
                    "error": str(e)
                })
        
        return {
            "fetched_boards": results,
            "success_count": len([r for r in results if "error" not in r])
        }
    
    def _get_cleaned_df(self, board_id: int) -> tuple:
        """Get cleaned DataFrame for a board."""
        if hasattr(self, '_last_fetched_data') and board_id in self._last_fetched_data:
            board_data = self._last_fetched_data[board_id]
        else:
            # Fetch if not in cache (though we shouldn't cache, this is just for the tool flow)
            client = get_monday_client()
            board_data = client.get_all_board_items(board_id)
            board_data["columns"] = client.get_board_columns(board_id)
        
        df = clean_board_data(board_data)
        return df, board_data
    
    def _handle_analyze_pipeline(self, params: Dict) -> Dict:
        """Analyze pipeline metrics."""
        board_id = params["board_id"]
        include_closed = params.get("include_closed", False)
        
        df, board_data = self._get_cleaned_df(board_id)
        analytics = BIAnalytics(df)
        
        result = analytics.compute_pipeline_value(
            exclude_closed=not include_closed
        )
        result["board_id"] = board_id
        result["board_name"] = board_data["board_name"]
        
        return result
    
    def _handle_analyze_conversions(self, params: Dict) -> Dict:
        """Analyze conversion rates."""
        board_id = params["board_id"]
        
        df, board_data = self._get_cleaned_df(board_id)
        analytics = BIAnalytics(df)
        
        result = analytics.compute_conversion_rates()
        result["board_id"] = board_id
        result["board_name"] = board_data["board_name"]
        
        return result
    
    def _handle_analyze_sector_breakdown(self, params: Dict) -> Dict:
        """Analyze sector breakdown."""
        board_id = params["board_id"]
        
        df, board_data = self._get_cleaned_df(board_id)
        analytics = BIAnalytics(df)
        
        result = analytics.compute_sector_breakdown()
        result["board_id"] = board_id
        result["board_name"] = board_data["board_name"]
        
        return result
    
    def _handle_analyze_quarterly_trends(self, params: Dict) -> Dict:
        """Analyze quarterly trends."""
        board_id = params["board_id"]
        
        df, board_data = self._get_cleaned_df(board_id)
        analytics = BIAnalytics(df)
        
        result = analytics.compute_quarterly_metrics()
        result["board_id"] = board_id
        result["board_name"] = board_data["board_name"]
        
        return result
    
    def _handle_compute_forecast(self, params: Dict) -> Dict:
        """Compute revenue forecast."""
        board_id = params["board_id"]
        
        df, board_data = self._get_cleaned_df(board_id)
        analytics = BIAnalytics(df)
        
        result = analytics.compute_forecast()
        result["board_id"] = board_id
        result["board_name"] = board_data["board_name"]
        
        return result
    
    def _handle_get_closed_revenue(self, params: Dict) -> Dict:
        """Get closed revenue with optional time filter."""
        board_id = params["board_id"]
        time_period = params.get("time_period")
        
        df, board_data = self._get_cleaned_df(board_id)
        analytics = BIAnalytics(df)
        
        if time_period:
            analytics = analytics.time_filter(relative_period=time_period)
        
        result = analytics.compute_closed_revenue()
        result["board_id"] = board_id
        result["board_name"] = board_data["board_name"]
        if time_period:
            result["time_period"] = time_period
        
        return result
    
    def _handle_cross_board_analysis(self, params: Dict) -> Dict:
        """Analyze multiple boards."""
        board_ids = params["board_ids"]
        
        from app.analytics import analyze_boards
        
        # Fetch all boards first
        client = get_monday_client()
        board_data_list = []
        
        for board_id in board_ids:
            try:
                board_data = client.get_all_board_items(board_id)
                board_data["columns"] = client.get_board_columns(board_id)
                board_data_list.append(board_data)
            except Exception as e:
                logger.error(f"Failed to fetch board {board_id}: {e}")
                continue
        
        result = analyze_boards(board_data_list)
        result["board_ids"] = board_ids
        
        return result
    
    def get_tool_traces(self) -> List[Dict]:
        """Get all tool call traces."""
        return self.tool_traces
    
    def clear_traces(self):
        """Clear tool traces."""
        self.tool_traces = []


# Singleton instance
_tool_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Get or create ToolRegistry singleton."""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry()
    return _tool_registry
