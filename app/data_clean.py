"""Data Cleaning and Normalization Layer - Handles messy Monday.com data."""

import re
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

logger = logging.getLogger(__name__)


class DataCleaner:
    """Cleans and normalizes messy data from Monday.com boards."""
    
    # Standard sector mappings for normalization
    SECTOR_MAPPINGS = {
        "tech": ["technology", "tech", "software", "saas", "it", "digital", "computing"],
        "healthcare": ["healthcare", "health", "medical", "pharma", "biotech", "life sciences"],
        "finance": ["finance", "financial", "banking", "fintech", "investment", "insurance"],
        "retail": ["retail", "e-commerce", "ecommerce", "consumer", "shopping"],
        "manufacturing": ["manufacturing", "industrial", "production", "factory"],
        "energy": ["energy", "oil", "gas", "renewable", "utilities", "power"],
        "consulting": ["consulting", "professional services", "advisory", "agency"],
        "education": ["education", "edtech", "training", "learning"],
        "real estate": ["real estate", "property", "construction", "housing"],
        "media": ["media", "entertainment", "content", "publishing", "marketing"],
    }
    
    # Stage normalization mappings
    STAGE_MAPPINGS = {
        "lead": ["lead", "prospect", "inquiry", "new", "cold", "suspect"],
        "qualified": ["qualified", "mql", "sql", "marketing qualified", "sales qualified"],
        "opportunity": ["opportunity", "demo", "meeting", "negotiation", "proposal"],
        "proposal": ["proposal", "quote", "pricing", "contract sent"],
        "closed_won": ["closed won", "won", "deal won", "closed", "customer", "signed"],
        "closed_lost": ["closed lost", "lost", "deal lost", "churned", "cancelled"],
    }
    
    def __init__(self):
        self.cleaning_log: List[Dict] = []
        logger.info("DataCleaner initialized")
    
    def _log_operation(self, operation: str, column: str, details: Dict):
        """Log a cleaning operation for traceability."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "column": column,
            "details": details
        }
        self.cleaning_log.append(log_entry)
        logger.info(f"Cleaning: {operation} on '{column}' - {details}")
    
    def clean_nulls(
        self, 
        df: pd.DataFrame, 
        columns: Optional[List[str]] = None,
        strategy: str = "drop"
    ) -> pd.DataFrame:
        """Clean null/empty values using specified strategy."""
        df = df.copy()
        target_cols = columns or df.columns.tolist()
        
        initial_count = len(df)
        null_counts_before = df[target_cols].isna().sum().sum() + (df[target_cols] == "").sum().sum()
        
        if strategy == "drop":
            # Drop rows with nulls in specified columns
            mask = pd.Series([True] * len(df))
            for col in target_cols:
                if col in df.columns:
                    mask &= df[col].notna() & (df[col] != "")
            df = df[mask].reset_index(drop=True)
            
        elif strategy == "fill":
            # Fill with appropriate defaults
            for col in target_cols:
                if col in df.columns:
                    if df[col].dtype in ['float64', 'int64']:
                        df[col] = df[col].fillna(0)
                    else:
                        df[col] = df[col].fillna("Unknown")
        
        null_counts_after = df[target_cols].isna().sum().sum() + (df[target_cols] == "").sum().sum()
        
        self._log_operation(
            "clean_nulls",
            ", ".join(target_cols),
            {
                "strategy": strategy,
                "rows_before": initial_count,
                "rows_after": len(df),
                "nulls_removed": null_counts_before - null_counts_after
            }
        )
        
        return df
    
    def normalize_revenue(
        self, 
        df: pd.DataFrame, 
        revenue_column: str,
        output_column: str = "revenue_normalized"
    ) -> pd.DataFrame:
        """Normalize revenue values from various formats to numeric."""
        df = df.copy()
        
        if revenue_column not in df.columns:
            logger.warning(f"Revenue column '{revenue_column}' not found")
            df[output_column] = np.nan
            return df
        
        def parse_revenue(val):
            if pd.isna(val) or val == "":
                return np.nan
            
            val_str = str(val).strip().lower()
            
            # Remove currency symbols and whitespace
            val_str = re.sub(r'[$€£¥,\s]', '', val_str)
            
            # Extract numeric value
            match = re.search(r'[-+]?[\d,]*\.?\d+', val_str)
            if not match:
                return np.nan
            
            num_str = match.group().replace(',', '')
            
            try:
                value = float(num_str)
            except ValueError:
                return np.nan
            
            # Handle multipliers
            if 'k' in val_str:
                value *= 1000
            elif 'm' in val_str:
                value *= 1000000
            elif 'b' in val_str:
                value *= 1000000000
            
            return value
        
        df[output_column] = df[revenue_column].apply(parse_revenue)
        
        valid_count = df[output_column].notna().sum()
        total_count = len(df)
        
        self._log_operation(
            "normalize_revenue",
            revenue_column,
            {
                "output_column": output_column,
                "valid_values": valid_count,
                "total_values": total_count,
                "success_rate": f"{valid_count/total_count*100:.1f}%"
            }
        )
        
        return df
    
    def standardize_sector(
        self, 
        df: pd.DataFrame, 
        sector_column: str,
        output_column: str = "sector_standardized"
    ) -> pd.DataFrame:
        """Standardize sector names using fuzzy matching."""
        df = df.copy()
        
        if sector_column not in df.columns:
            logger.warning(f"Sector column '{sector_column}' not found")
            df[output_column] = "Unknown"
            return df
        
        def match_sector(val):
            if pd.isna(val) or val == "":
                return "Unknown"
            
            val_str = str(val).strip().lower()
            
            # Direct match first
            for standard, variants in self.SECTOR_MAPPINGS.items():
                if val_str in variants:
                    return standard.title()
            
            # Fuzzy match
            best_match = None
            best_score = 0
            
            for standard, variants in self.SECTOR_MAPPINGS.items():
                for variant in variants:
                    score = fuzz.partial_ratio(val_str, variant)
                    if score > best_score and score >= 70:
                        best_score = score
                        best_match = standard
            
            if best_match:
                return best_match.title()
            
            return "Other"
        
        df[output_column] = df[sector_column].apply(match_sector)
        
        sector_counts = df[output_column].value_counts().to_dict()
        
        self._log_operation(
            "standardize_sector",
            sector_column,
            {
                "output_column": output_column,
                "unique_sectors": len(sector_counts),
                "sector_distribution": sector_counts
            }
        )
        
        return df
    
    def normalize_stage(
        self, 
        df: pd.DataFrame, 
        stage_column: str,
        output_column: str = "stage_normalized"
    ) -> pd.DataFrame:
        """Normalize deal/sales stage values."""
        df = df.copy()
        
        if stage_column not in df.columns:
            logger.warning(f"Stage column '{stage_column}' not found")
            df[output_column] = "Unknown"
            return df
        
        def match_stage(val):
            if pd.isna(val) or val == "":
                return "Unknown"
            
            val_str = str(val).strip().lower()
            
            # Direct match
            for standard, variants in self.STAGE_MAPPINGS.items():
                if val_str in variants:
                    return standard
            
            # Fuzzy match
            best_match = None
            best_score = 0
            
            for standard, variants in self.STAGE_MAPPINGS.items():
                for variant in variants:
                    score = fuzz.ratio(val_str, variant)
                    if score > best_score and score >= 75:
                        best_score = score
                        best_match = standard
            
            if best_match:
                return best_match
            
            # Check if it contains keywords
            if any(x in val_str for x in ["win", "sign", "close", "won"]):
                return "closed_won"
            if any(x in val_str for x in ["lose", "lost", "reject", "no deal"]):
                return "closed_lost"
            
            return "other"
        
        df[output_column] = df[stage_column].apply(match_stage)
        
        stage_counts = df[output_column].value_counts().to_dict()
        
        self._log_operation(
            "normalize_stage",
            stage_column,
            {
                "output_column": output_column,
                "stage_distribution": stage_counts
            }
        )
        
        return df
    
    def parse_dates(
        self, 
        df: pd.DataFrame, 
        date_column: str,
        output_column: str = "date_parsed"
    ) -> pd.DataFrame:
        """Parse various date formats into datetime objects."""
        df = df.copy()
        
        if date_column not in df.columns:
            logger.warning(f"Date column '{date_column}' not found")
            df[output_column] = pd.NaT
            return df
        
        def try_parse_date(val):
            if pd.isna(val) or val == "":
                return pd.NaT
            
            if isinstance(val, datetime):
                return val
            
            val_str = str(val).strip()
            
            # Common date formats to try
            formats = [
                "%Y-%m-%d",
                "%Y-%m-%d %H:%M:%S",
                "%m/%d/%Y",
                "%d/%m/%Y",
                "%m-%d-%Y",
                "%d-%m-%Y",
                "%b %d, %Y",
                "%B %d, %Y",
                "%d %b %Y",
                "%d %B %Y",
                "%Y%m%d",
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(val_str, fmt)
                except ValueError:
                    continue
            
            # Try pandas parser as fallback
            try:
                return pd.to_datetime(val_str)
            except:
                return pd.NaT
        
        df[output_column] = df[date_column].apply(try_parse_date)
        
        valid_dates = df[output_column].notna().sum()
        total = len(df)
        
        self._log_operation(
            "parse_dates",
            date_column,
            {
                "output_column": output_column,
                "valid_dates": valid_dates,
                "total": total,
                "success_rate": f"{valid_dates/total*100:.1f}%"
            }
        )
        
        return df
    
    def add_time_features(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """Add year, quarter, month features from date column."""
        df = df.copy()
        
        if date_column not in df.columns:
            return df
        
        # Ensure column is datetime type
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            logger.warning(f"Column '{date_column}' is not datetime type, skipping time features")
            return df
        
        df["year"] = df[date_column].dt.year
        df["quarter"] = df[date_column].dt.quarter
        df["month"] = df[date_column].dt.month
        df["month_name"] = df[date_column].dt.strftime("%b")
        df["quarter_label"] = df[date_column].dt.year.astype(str) + "-Q" + df[date_column].dt.quarter.astype(str)
        
        self._log_operation(
            "add_time_features",
            date_column,
            {"features_added": ["year", "quarter", "month", "month_name", "quarter_label"]}
        )
        
        return df
    
    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names (lowercase, no spaces, no special chars)."""
        df = df.copy()
        
        original_names = df.columns.tolist()
        new_names = []
        
        for col in df.columns:
            # Convert to lowercase
            new_name = col.lower()
            # Replace spaces and special chars with underscore
            new_name = re.sub(r'[^\w]', '_', new_name)
            # Remove multiple underscores
            new_name = re.sub(r'_+', '_', new_name)
            # Remove leading/trailing underscores
            new_name = new_name.strip('_')
            new_names.append(new_name)
        
        df.columns = new_names
        
        # Log renames
        renames = {o: n for o, n in zip(original_names, new_names) if o != n}
        if renames:
            self._log_operation(
                "clean_column_names",
                "all",
                {"renamed": renames}
            )
        
        return df
    
    def get_cleaning_report(self) -> Dict:
        """Get a summary of all cleaning operations performed."""
        return {
            "total_operations": len(self.cleaning_log),
            "operations": self.cleaning_log
        }
    
    def full_clean_pipeline(
        self,
        df: pd.DataFrame,
        revenue_col: Optional[str] = None,
        sector_col: Optional[str] = None,
        stage_col: Optional[str] = None,
        date_col: Optional[str] = None
    ) -> pd.DataFrame:
        """Execute full cleaning pipeline on raw board data."""
        logger.info("Starting full cleaning pipeline")
        
        # Reset log
        self.cleaning_log = []
        
        # Clean column names first
        df = self.clean_column_names(df)
        
        # Clean nulls
        df = self.clean_nulls(df, strategy="fill")
        
        # Normalize revenue if specified
        if revenue_col:
            df = self.normalize_revenue(df, revenue_col)
        
        # Standardize sector if specified
        if sector_col:
            df = self.standardize_sector(df, sector_col)
        
        # Normalize stage if specified
        if stage_col:
            df = self.normalize_stage(df, stage_col)
        
        # Parse dates if specified
        if date_col:
            df = self.parse_dates(df, date_col)
            df = self.add_time_features(df, f"{date_col}_parsed" if f"{date_col}_parsed" in df.columns else date_col)
        
        logger.info(f"Cleaning pipeline complete. Final shape: {df.shape}")
        return df


def clean_board_data(board_data: Dict, column_mapping: Optional[Dict] = None) -> pd.DataFrame:
    """Convenience function to clean board data with common mappings."""
    from app.monday_client import MondayClient
    
    try:
        client = MondayClient()
        df = client.items_to_dataframe(board_data)
        
        if df.empty:
            logger.warning("No data found in board")
            return pd.DataFrame()
        
        cleaner = DataCleaner()
        
        # First, clean column names to get snake_case versions
        df = cleaner.clean_column_names(df)
        cols_lower = {c: c for c in df.columns}  # already lowercase from cleaning
        logger.info(f"Cleaned columns: {list(df.columns)}")
        
        # Auto-detect columns using priority scoring on cleaned names
        # Revenue column - look for deal + value, or amount/price patterns
        revenue_col = None
        revenue_scores = {}
        for col in cols_lower:
            score = 0
            if 'deal' in col and 'value' in col:
                score = 100  # Best match: contains both deal and value
            elif 'amount' in col:
                score = 80
            elif 'revenue' in col:
                score = 70
            elif 'price' in col or 'budget' in col:
                score = 60
            elif 'value' in col:
                score = 50  # Generic value column
            if score > 0:
                revenue_scores[col] = score
        
        if revenue_scores:
            revenue_col = max(revenue_scores, key=revenue_scores.get)
            logger.info(f"Detected revenue column: {revenue_col} (score: {revenue_scores[revenue_col]})")
        
        # Stage column - look for stage patterns first, avoid status unless it's clearly stage
        stage_col = None
        stage_scores = {}
        for col in cols_lower:
            score = 0
            if 'stage' in col:
                score = 100  # Best match
            elif 'pipeline' in col or 'phase' in col:
                score = 80
            elif col == 'status':  # Only match exact 'status', not 'deal_status'
                score = 40
            if score > 0:
                stage_scores[col] = score
        
        if stage_scores:
            stage_col = max(stage_scores, key=stage_scores.get)
            logger.info(f"Detected stage column: {stage_col} (score: {stage_scores[stage_col]})")
        
        # Sector column
        sector_col = None
        sector_scores = {}
        for col in cols_lower:
            score = 0
            if 'sector' in col:
                score = 100
            elif 'industry' in col:
                score = 80
            elif 'vertical' in col:
                score = 70
            elif 'category' in col:
                score = 60
            if score > 0:
                sector_scores[col] = score
        
        if sector_scores:
            sector_col = max(sector_scores, key=sector_scores.get)
            logger.info(f"Detected sector column: {sector_col} (score: {sector_scores[sector_col]})")
        
        # Date column
        date_col = None
        for col in cols_lower:
            if col == 'created_at' or col == 'updated_at':
                date_col = col
                logger.info(f"Detected date column: {date_col}")
                break
            elif 'close_date' in col or 'date' in col:
                if not date_col:  # Prefer close_date over generic date
                    date_col = col
        
        if date_col and not (date_col == 'created_at' or date_col == 'updated_at'):
            logger.info(f"Detected date column: {date_col}")
        
        # Use provided mapping if available (overrides auto-detection)
        if column_mapping:
            if column_mapping.get("revenue"):
                revenue_col = column_mapping.get("revenue")
                logger.info(f"Using mapped revenue column: {revenue_col}")
            if column_mapping.get("stage"):
                stage_col = column_mapping.get("stage")
                logger.info(f"Using mapped stage column: {stage_col}")
            if column_mapping.get("sector"):
                sector_col = column_mapping.get("sector")
            if column_mapping.get("date"):
                date_col = column_mapping.get("date")
        
        return cleaner.full_clean_pipeline(df, revenue_col, sector_col, stage_col, date_col)
        
    except Exception as e:
        logger.error(f"Error cleaning board data: {str(e)}")
        raise ValueError(f"Failed to clean board data: {str(e)}")
