"""Test script to diagnose Monday.com data extraction issues."""

import os
import sys
import json
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.monday_client import get_monday_client
from app.data_clean import clean_board_data

def test_board_data(board_id):
    """Test fetching and cleaning data from a board."""
    print(f"\n{'='*60}")
    print(f"Testing Board ID: {board_id}")
    print(f"{'='*60}")
    
    try:
        client = get_monday_client()
        
        # Step 1: Fetch board data
        print("\n1. Fetching board data...")
        board_data = client.get_all_board_items(board_id)
        print(f"   Board name: {board_data.get('board_name', 'Unknown')}")
        print(f"   Total items: {board_data.get('total_count', 0)}")
        
        # Step 2: Fetch columns
        print("\n2. Fetching column definitions...")
        columns = client.get_board_columns(board_id)
        print(f"   Total columns: {len(columns)}")
        print(f"   Columns: {[c['title'] for c in columns]}")
        
        # Add columns to board_data for cleaning
        board_data["columns"] = columns
        
        # Step 3: Show sample raw item
        print("\n3. Sample raw item (first item):")
        items = board_data.get("items", [])
        if items:
            first_item = items[0]
            print(f"   Item name: {first_item.get('name', 'N/A')}")
            print(f"   Column values:")
            for col_val in first_item.get("column_values", []):
                col_title = next((c['title'] for c in columns if c['id'] == col_val['id']), col_val['id'])
                text = col_val.get('text', '')
                value = col_val.get('value', '')
                print(f"     - {col_title}: text='{text}', value='{value}'")
        
        # Step 4: Convert to DataFrame
        print("\n4. Converting to DataFrame...")
        df = client.items_to_dataframe(board_data)
        print(f"   DataFrame shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        
        # Step 5: Show sample DataFrame row
        print("\n5. Sample DataFrame row (first row):")
        if not df.empty:
            first_row = df.iloc[0].to_dict()
            for key, val in list(first_row.items())[:10]:
                print(f"     - {key}: {val}")
        
        # Step 6: Try cleaning
        print("\n6. Running data cleaning...")
        df_clean = clean_board_data(board_data)
        print(f"   Cleaned DataFrame shape: {df_clean.shape}")
        print(f"   Cleaned columns: {list(df_clean.columns)}")
        
        # Step 7: Check for normalized columns
        print("\n7. Checking for normalized columns:")
        normalized_cols = [c for c in df_clean.columns if 'normalized' in c.lower() or 'standardized' in c.lower()]
        print(f"   Normalized columns found: {normalized_cols}")
        
        # Step 8: Check revenue data
        if 'revenue_normalized' in df_clean.columns:
            revenue_vals = df_clean['revenue_normalized'].dropna()
            print(f"\n8. Revenue data:")
            print(f"   Non-null values: {len(revenue_vals)}")
            print(f"   Sample values: {revenue_vals.head(3).tolist()}")
        else:
            print("\n8. No 'revenue_normalized' column found!")
        
        # Step 9: Check stage data
        if 'stage_normalized' in df_clean.columns:
            stage_counts = df_clean['stage_normalized'].value_counts()
            print(f"\n9. Stage distribution:")
            print(f"   {stage_counts.to_dict()}")
        else:
            print("\n9. No 'stage_normalized' column found!")
        
        print(f"\n{'='*60}")
        print("TEST COMPLETED SUCCESSFULLY")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"{'='*60}\n")

if __name__ == "__main__":
    # Test with the board ID from the error
    board_id = 5026873403
    test_board_data(board_id)
