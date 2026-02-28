"""BI Analytics Engine - Computes pipeline metrics, conversions, breakdowns."""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class BIAnalytics:
    """Business Intelligence analytics engine for pipeline and sales data."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.analysis_log: List[Dict] = []
        logger.info(f"BIAnalytics initialized with {len(df)} records")
    
    def _log_analysis(self, metric: str, result: Dict):
        """Log analysis operation."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "metric": metric,
            "result": result
        }
        self.analysis_log.append(log_entry)
        logger.info(f"Analysis: {metric} = {result}")
    
    def compute_pipeline_value(
        self, 
        revenue_col: str = "revenue_normalized",
        stage_col: str = "stage_normalized",
        exclude_closed: bool = True
    ) -> Dict:
        """Compute total pipeline value by stage."""
        if revenue_col not in self.df.columns:
            return {"error": f"Revenue column '{revenue_col}' not found"}
        
        df = self.df.copy()
        
        # Filter out closed deals if requested
        if exclude_closed and stage_col in df.columns:
            df = df[~df[stage_col].isin(["closed_won", "closed_lost"])]
        
        total_pipeline = df[revenue_col].sum()
        deal_count = len(df)
        avg_deal_size = df[revenue_col].mean() if deal_count > 0 else 0
        
        # Breakdown by stage
        stage_breakdown = {}
        if stage_col in df.columns:
            stage_breakdown = df.groupby(stage_col)[revenue_col].agg(['sum', 'count', 'mean']).to_dict('index')
        
        result = {
            "total_pipeline_value": round(total_pipeline, 2),
            "deal_count": deal_count,
            "average_deal_size": round(avg_deal_size, 2),
            "stage_breakdown": stage_breakdown,
            "currency": "USD"
        }
        
        self._log_analysis("pipeline_value", result)
        return result
    
    def compute_closed_revenue(
        self,
        revenue_col: str = "revenue_normalized",
        stage_col: str = "stage_normalized"
    ) -> Dict:
        """Compute closed won and lost revenue."""
        if revenue_col not in self.df.columns or stage_col not in self.df.columns:
            return {"error": f"Required columns not found"}
        
        df = self.df.copy()
        
        # Won deals
        won_df = df[df[stage_col] == "closed_won"]
        won_revenue = won_df[revenue_col].sum()
        won_count = len(won_df)
        
        # Lost deals
        lost_df = df[df[stage_col] == "closed_lost"]
        lost_revenue = lost_df[revenue_col].sum()
        lost_count = len(lost_df)
        
        # Win rate
        total_closed = won_count + lost_count
        win_rate = (won_count / total_closed * 100) if total_closed > 0 else 0
        
        result = {
            "closed_won": {
                "revenue": round(won_revenue, 2),
                "count": won_count
            },
            "closed_lost": {
                "revenue": round(lost_revenue, 2),
                "count": lost_count
            },
            "win_rate_percent": round(win_rate, 2),
            "total_closed_deals": total_closed
        }
        
        self._log_analysis("closed_revenue", result)
        return result
    
    def compute_conversion_rates(
        self,
        stage_col: str = "stage_normalized"
    ) -> Dict:
        """Compute conversion rates between pipeline stages."""
        if stage_col not in self.df.columns:
            return {"error": f"Stage column '{stage_col}' not found"}
        
        stage_counts = self.df[stage_col].value_counts().to_dict()
        total = len(self.df)
        
        # Define funnel order
        funnel_order = ["lead", "qualified", "opportunity", "proposal", "closed_won"]
        
        conversion_rates = {}
        prev_count = total
        
        for stage in funnel_order:
            if stage in stage_counts:
                current_count = stage_counts[stage]
                rate = (current_count / prev_count * 100) if prev_count > 0 else 0
                conversion_rates[f"to_{stage}"] = {
                    "count": current_count,
                    "conversion_rate": round(rate, 2),
                    "percent_of_total": round(current_count / total * 100, 2)
                }
                prev_count = current_count
        
        result = {
            "stage_counts": stage_counts,
            "conversion_rates": conversion_rates,
            "total_deals": total
        }
        
        self._log_analysis("conversion_rates", result)
        return result
    
    def compute_sector_breakdown(
        self,
        sector_col: str = "sector_standardized",
        revenue_col: str = "revenue_normalized",
        stage_col: str = "stage_normalized"
    ) -> Dict:
        """Compute metrics broken down by sector."""
        if sector_col not in self.df.columns:
            return {"error": f"Sector column '{sector_col}' not found"}
        
        df = self.df.copy()
        
        # Basic sector stats
        sector_stats = df.groupby(sector_col).agg({
            revenue_col: ['sum', 'mean', 'count'] if revenue_col in df.columns else 'count',
            stage_col: lambda x: (x == 'closed_won').sum() if stage_col in df.columns else 0
        })
        
        # Flatten column names
        if revenue_col in df.columns and stage_col in df.columns:
            sector_stats.columns = ['total_revenue', 'avg_deal_size', 'deal_count', 'won_count']
            sector_stats['win_rate'] = (sector_stats['won_count'] / sector_stats['deal_count'] * 100).round(2)
        else:
            sector_stats.columns = ['deal_count']
        
        # Convert to dict
        breakdown = sector_stats.to_dict('index')
        
        # Top sectors by revenue
        top_sectors = sorted(
            breakdown.items(),
            key=lambda x: x[1].get('total_revenue', 0),
            reverse=True
        )[:5]
        
        result = {
            "total_sectors": len(breakdown),
            "sector_breakdown": breakdown,
            "top_sectors_by_revenue": dict(top_sectors)
        }
        
        self._log_analysis("sector_breakdown", result)
        return result
    
    def compute_quarterly_metrics(
        self,
        date_col: str = "date_parsed",
        revenue_col: str = "revenue_normalized",
        stage_col: str = "stage_normalized"
    ) -> Dict:
        """Compute quarterly grouped metrics."""
        if date_col not in self.df.columns:
            return {"error": f"Date column '{date_col}' not found"}
        
        df = self.df.copy()
        df = df[df[date_col].notna()]
        
        if len(df) == 0:
            return {"error": "No valid dates found"}
        
        # Ensure we have quarter_label
        if "quarter_label" not in df.columns:
            df["quarter_label"] = df[date_col].dt.year.astype(str) + "-Q" + df[date_col].dt.quarter.astype(str)
        
        # Quarterly aggregation
        agg_dict = {"item_id": "count"}
        if revenue_col in df.columns:
            agg_dict[revenue_col] = "sum"
        if stage_col in df.columns:
            agg_dict[stage_col] = lambda x: (x == "closed_won").sum()
        
        quarterly = df.groupby("quarter_label").agg(agg_dict)
        
        if revenue_col in df.columns and stage_col in df.columns:
            quarterly.columns = ['deal_count', 'revenue', 'won_deals']
        elif revenue_col in df.columns:
            quarterly.columns = ['deal_count', 'revenue']
        else:
            quarterly.columns = ['deal_count']
        
        quarterly_dict = quarterly.to_dict('index')
        
        # Calculate QoQ growth
        quarters = sorted(quarterly_dict.keys())
        qoq_growth = {}
        
        for i, q in enumerate(quarters[1:], 1):
            prev_q = quarters[i-1]
            curr_revenue = quarterly_dict[q].get('revenue', 0)
            prev_revenue = quarterly_dict[prev_q].get('revenue', 0)
            
            if prev_revenue > 0:
                growth = ((curr_revenue - prev_revenue) / prev_revenue) * 100
                qoq_growth[f"{prev_q}_to_{q}"] = round(growth, 2)
        
        result = {
            "quarterly_data": quarterly_dict,
            "quarter_over_quarter_growth": qoq_growth,
            "quarters_covered": quarters
        }
        
        self._log_analysis("quarterly_metrics", result)
        return result
    
    def compute_forecast(
        self,
        revenue_col: str = "revenue_normalized",
        stage_col: str = "stage_normalized",
        probability_map: Optional[Dict[str, float]] = None
    ) -> Dict:
        """Compute weighted forecast based on stage probabilities."""
        if revenue_col not in self.df.columns or stage_col not in self.df.columns:
            return {"error": "Required columns not found"}
        
        # Default stage probabilities
        default_probs = {
            "lead": 0.10,
            "qualified": 0.25,
            "opportunity": 0.40,
            "proposal": 0.60,
            "closed_won": 1.0,
            "closed_lost": 0.0
        }
        
        probs = probability_map or default_probs
        df = self.df.copy()
        
        # Calculate weighted revenue
        df["win_probability"] = df[stage_col].map(probs).fillna(0.1)
        df["weighted_revenue"] = df[revenue_col] * df["win_probability"]
        
        total_weighted = df["weighted_revenue"].sum()
        total_unweighted = df[revenue_col].sum()
        
        # By stage forecast
        stage_forecast = df.groupby(stage_col).agg({
            revenue_col: 'sum',
            "weighted_revenue": 'sum',
            "item_id": 'count'
        }).to_dict('index')
        
        result = {
            "total_weighted_forecast": round(total_weighted, 2),
            "total_pipeline": round(total_unweighted, 2),
            "forecast_coverage": round(total_weighted / total_unweighted * 100, 2) if total_unweighted > 0 else 0,
            "stage_forecast": stage_forecast,
            "probabilities_used": probs
        }
        
        self._log_analysis("forecast", result)
        return result
    
    def time_filter(
        self,
        date_col: str = "date_parsed",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        relative_period: Optional[str] = None
    ) -> "BIAnalytics":
        """Filter data by time period."""
        if date_col not in self.df.columns:
            logger.warning(f"Date column '{date_col}' not found, skipping time filter")
            return self
        
        df = self.df.copy()
        df = df[df[date_col].notna()]
        
        # Handle relative periods
        if relative_period:
            now = datetime.now()
            if relative_period == "this_month":
                start_date = now.replace(day=1)
                end_date = now
            elif relative_period == "last_month":
                first_of_month = now.replace(day=1)
                end_date = first_of_month - timedelta(days=1)
                start_date = end_date.replace(day=1)
            elif relative_period == "this_quarter":
                quarter = (now.month - 1) // 3
                start_date = now.replace(month=quarter * 3 + 1, day=1)
                end_date = now
            elif relative_period == "last_quarter":
                quarter = (now.month - 1) // 3
                if quarter == 0:
                    start_date = now.replace(year=now.year - 1, month=10, day=1)
                    end_date = now.replace(year=now.year - 1, month=12, day=31)
                else:
                    start_date = now.replace(month=(quarter - 1) * 3 + 1, day=1)
                    end_date = now.replace(month=quarter * 3 + 1, day=1) - timedelta(days=1)
            elif relative_period == "ytd":
                start_date = now.replace(month=1, day=1)
                end_date = now
            elif relative_period == "last_30_days":
                start_date = now - timedelta(days=30)
                end_date = now
            elif relative_period == "last_90_days":
                start_date = now - timedelta(days=90)
                end_date = now
        
        # Apply filters
        if start_date:
            df = df[df[date_col] >= start_date]
        if end_date:
            df = df[date_col] <= end_date
        
        logger.info(f"Time filter applied: {len(df)} records remaining")
        return BIAnalytics(df)
    
    def cross_board_join(
        self,
        other_df: pd.DataFrame,
        join_key: str,
        how: str = "outer"
    ) -> "BIAnalytics":
        """Join data from multiple boards for cross-board analysis."""
        merged = pd.merge(
            self.df,
            other_df,
            on=join_key,
            how=how,
            suffixes=('', '_other')
        )
        
        logger.info(f"Cross-board join: {len(self.df)} + {len(other_df)} = {len(merged)} records")
        return BIAnalytics(merged)
    
    def get_top_deals(
        self,
        n: int = 10,
        revenue_col: str = "revenue_normalized"
    ) -> List[Dict]:
        """Get top N deals by revenue."""
        if revenue_col not in self.df.columns:
            return []
        
        top = self.df.nlargest(n, revenue_col)
        return top.to_dict('records')
    
    def get_summary_stats(self) -> Dict:
        """Get overall summary statistics."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        stats = {
            "total_records": len(self.df),
            "columns": self.df.columns.tolist(),
            "numeric_columns": numeric_cols
        }
        
        if numeric_cols:
            stats["numeric_summary"] = self.df[numeric_cols].describe().to_dict()
        
        return stats
    
    def full_analysis(
        self,
        revenue_col: str = "revenue_normalized",
        stage_col: str = "stage_normalized",
        sector_col: str = "sector_standardized",
        date_col: str = "date_parsed"
    ) -> Dict:
        """Run complete analysis suite."""
        logger.info("Running full analysis suite")
        
        results = {
            "summary": self.get_summary_stats(),
            "pipeline": self.compute_pipeline_value(revenue_col, stage_col),
            "closed_revenue": self.compute_closed_revenue(revenue_col, stage_col),
            "conversions": self.compute_conversion_rates(stage_col),
        }
        
        if sector_col in self.df.columns:
            results["sector_breakdown"] = self.compute_sector_breakdown(sector_col, revenue_col, stage_col)
        
        if date_col in self.df.columns:
            results["quarterly"] = self.compute_quarterly_metrics(date_col, revenue_col, stage_col)
        
        results["forecast"] = self.compute_forecast(revenue_col, stage_col)
        results["top_deals"] = self.get_top_deals(5, revenue_col)
        
        return results
    
    def get_analysis_trace(self) -> List[Dict]:
        """Get trace of all analysis operations."""
        return self.analysis_log


def analyze_boards(board_data_list: List[Dict], join_key: Optional[str] = None) -> Dict:
    """Analyze multiple boards, optionally joining them."""
    from data_clean import clean_board_data
    
    all_results = []
    
    for board_data in board_data_list:
        try:
            df = clean_board_data(board_data)
            analytics = BIAnalytics(df)
            results = analytics.full_analysis()
            results["board_id"] = board_data["board_id"]
            results["board_name"] = board_data["board_name"]
            all_results.append(results)
        except Exception as e:
            logger.error(f"Failed to analyze board {board_data.get('board_id')}: {str(e)}")
            continue
    
    # Aggregate across boards
    if len(all_results) > 1:
        total_pipeline = sum(r["pipeline"].get("total_pipeline_value", 0) for r in all_results)
        total_won = sum(r["closed_revenue"].get("closed_won", {}).get("revenue", 0) for r in all_results)
        
        return {
            "boards_analyzed": len(all_results),
            "individual_results": all_results,
            "cross_board_summary": {
                "total_pipeline_value": round(total_pipeline, 2),
                "total_closed_won": round(total_won, 2)
            }
        }
    
    return all_results[0] if all_results else {"error": "No boards could be analyzed"}
