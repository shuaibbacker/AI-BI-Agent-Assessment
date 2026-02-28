"""Monday.com API Client - Live data fetcher with full logging, no caching."""

import os
import json
import logging
import requests
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MondayAPIError(Exception):
    """Custom exception for Monday.com API errors."""
    pass


class MondayClient:
    """Client for Monday.com API - fetches live data with full logging, no caching."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("MONDAY_API_KEY")
        if not self.api_key:
            raise MondayAPIError("Monday.com API key not found")
        
        self.base_url = "https://api.monday.com/v2"
        self.headers = {
            "Authorization": self.api_key,
            "Content-Type": "application/json"
        }
        logger.info("MondayClient initialized")
    
    def _execute_query(self, query: str, variables: Optional[Dict] = None) -> Dict:
        """Execute GraphQL query with full logging."""
        payload = {"query": query}
        if variables:
            payload["variables"] = variables
        
        logger.info(f"API CALL - Query: {query[:100]}...")
        logger.info(f"API CALL - Variables: {variables}")
        
        try:
            response = requests.post(
                self.base_url,
                json=payload,
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            if "errors" in data:
                error_msg = str(data["errors"])
                logger.error(f"API ERROR: {error_msg}")
                raise MondayAPIError(f"Monday API returned errors: {error_msg}")
            
            logger.info(f"API SUCCESS - Response keys: {list(data.keys())}")
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API REQUEST FAILED: {str(e)}")
            raise MondayAPIError(f"Request failed: {str(e)}")
    
    def get_boards(self, limit: int = 10) -> List[Dict]:
        """Fetch all accessible boards."""
        query = f"""
        query {{
            boards(limit: {limit}) {{
                id
                name
                description
                state
            }}
        }}
        """
        
        data = self._execute_query(query)
        boards = data.get("data", {}).get("boards", [])
        logger.info(f"Fetched {len(boards)} boards")
        return boards
    
    def get_board_columns(self, board_id: int) -> List[Dict]:
        """Fetch column definitions for a board."""
        query = f"""
        query {{
            boards(ids: {board_id}) {{
                columns {{
                    id
                    title
                    type
                    settings_str
                }}
            }}
        }}
        """
        
        data = self._execute_query(query)
        columns = data.get("data", {}).get("boards", [{}])[0].get("columns", [])
        logger.info(f"Fetched {len(columns)} columns for board {board_id}")
        return columns
    
    def get_board_items(
        self, 
        board_id: int, 
        limit: int = 100,
        cursor: Optional[str] = None
    ) -> Dict:
        """Fetch items from a specific board with pagination support."""
        
        # Build the items_page query with optional cursor
        cursor_part = f', cursor: "{cursor}"' if cursor else ""
        
        query = f"""
        query {{
            boards(ids: {board_id}) {{
                id
                name
                items_page(limit: {limit}{cursor_part}) {{
                    cursor
                    items {{
                        id
                        name
                        created_at
                        updated_at
                        column_values {{
                            id
                            text
                            value
                            type
                        }}
                    }}
                }}
            }}
        }}
        """
        
        data = self._execute_query(query)
        board_data = data.get("data", {}).get("boards", [{}])[0]
        items_page = board_data.get("items_page", {})
        
        items = items_page.get("items", [])
        next_cursor = items_page.get("cursor")
        
        logger.info(f"Fetched {len(items)} items from board {board_id}")
        if next_cursor:
            logger.info(f"Pagination cursor available for board {board_id}")
        
        return {
            "board_id": board_id,
            "board_name": board_data.get("name", ""),
            "items": items,
            "cursor": next_cursor,
            "has_more": next_cursor is not None
        }
    
    def get_all_board_items(self, board_id: int) -> Dict:
        """Fetch all items from a board (handles pagination)."""
        all_items = []
        cursor = None
        page_count = 0
        
        while True:
            page_count += 1
            result = self.get_board_items(board_id, limit=100, cursor=cursor)
            all_items.extend(result["items"])
            
            if not result["has_more"]:
                break
            
            cursor = result["cursor"]
            logger.info(f"Fetching page {page_count + 1} for board {board_id}")
        
        logger.info(f"Fetched {len(all_items)} total items across {page_count} pages")
        return {
            "board_id": board_id,
            "board_name": result["board_name"],
            "items": all_items,
            "total_count": len(all_items)
        }
    
    def fetch_multiple_boards(self, board_ids: List[int]) -> List[Dict]:
        """Fetch data from multiple boards for cross-board analysis."""
        results = []
        for board_id in board_ids:
            try:
                board_data = self.get_all_board_items(board_id)
                columns = self.get_board_columns(board_id)
                board_data["columns"] = columns
                results.append(board_data)
            except MondayAPIError as e:
                logger.error(f"Failed to fetch board {board_id}: {str(e)}")
                continue
        
        logger.info(f"Successfully fetched {len(results)} boards out of {len(board_ids)}")
        return results
    
    def items_to_dataframe(self, board_data: Dict) -> pd.DataFrame:
        """Convert board items to pandas DataFrame with extracted column values."""
        items = board_data.get("items", [])
        columns = board_data.get("columns", [])
        
        # Create column mapping: id -> title and id -> type
        column_map = {col["id"]: col["title"] for col in columns}
        column_type_map = {col["id"]: col.get("type", "") for col in columns}
        
        rows = []
        for item in items:
            row = {
                "item_id": item["id"],
                "item_name": item["name"],
                "created_at": item.get("created_at"),
                "updated_at": item.get("updated_at"),
                "board_id": board_data["board_id"],
                "board_name": board_data["board_name"]
            }
            
            # Add column values - extract from both text and value fields
            for col_val in item.get("column_values", []):
                col_id = col_val["id"]
                col_title = column_map.get(col_id, col_id)
                col_type = column_type_map.get(col_id, "")
                
                # Get text value (fallback)
                text_value = col_val.get("text", "") or ""
                
                # Try to extract from JSON value field for richer data
                json_value = col_val.get("value")
                if json_value and json_value != "{}":
                    try:
                        import json
                        parsed = json.loads(json_value)
                        # Extract based on column type
                        if col_type == "numbers":
                            text_value = str(parsed.get("number", text_value))
                        elif col_type == "dropdown":
                            labels = parsed.get("labels", [])
                            if labels:
                                text_value = ", ".join([l.get("name", "") for l in labels])
                        elif col_type == "status" or col_type == "label":
                            text_value = parsed.get("label", text_value) or text_value
                        elif col_type == "date":
                            text_value = parsed.get("date", text_value) or text_value
                        elif col_type == "people":
                            persons = parsed.get("personsAndTeams", [])
                            if persons:
                                text_value = ", ".join([p.get("name", "") for p in persons])
                    except (json.JSONDecodeError, AttributeError):
                        pass
                
                row[col_title] = text_value
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        logger.info(f"Converted to DataFrame: {df.shape[0]} rows, {df.shape[1]} columns")
        logger.info(f"Columns found: {list(df.columns)}")
        return df


# Singleton instance for reuse
_client_instance: Optional[MondayClient] = None


def get_monday_client() -> MondayClient:
    """Get or create MondayClient singleton."""
    global _client_instance
    if _client_instance is None:
        _client_instance = MondayClient()
    return _client_instance
