"""Streamlit Frontend - Chat UI for the BI Agent."""

import os
import json
import requests
import streamlit as st
from datetime import datetime
from typing import Dict, List, Optional

# Page configuration
st.set_page_config(
    page_title="BI Agent - Monday.com Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if "tool_trace" not in st.session_state:
        st.session_state.tool_trace = []
    
    if "show_trace" not in st.session_state:
        st.session_state.show_trace = True
    
    if "boards_cache" not in st.session_state:
        st.session_state.boards_cache = []


def send_query(query: str, session_id: str) -> Dict:
    """Send query to API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={"query": query, "session_id": session_id},
            timeout=120
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API. Is the backend running?"}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. The query may be too complex."}
    except Exception as e:
        return {"error": str(e)}


def fetch_boards() -> List[Dict]:
    """Fetch available boards."""
    try:
        response = requests.get(f"{API_BASE_URL}/boards", timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("boards", [])
    except:
        return []


def analyze_board(board_id: int, analysis_type: str = "full") -> Dict:
    """Analyze a specific board."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/analyze",
            json={"board_id": board_id, "analysis_type": analysis_type},
            timeout=120
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def render_tool_trace(trace: List[Dict]):
    """Render tool execution trace in an expandable section."""
    if not trace:
        return
    
    with st.expander("🔍 Tool Execution Trace", expanded=False):
        for i, entry in enumerate(trace, 1):
            st.markdown(f"**Step {i}: {entry['tool']}**")
            st.markdown(f"*Time: {entry['timestamp']}*")
            
            # Show parameters
            if entry.get('parameters'):
                st.json(entry['parameters'])
            
            # Show result summary
            if entry.get('result'):
                result = entry['result']
                if isinstance(result, dict):
                    if 'error' in result:
                        st.error(f"Error: {result['error']}")
                    else:
                        # Show key metrics
                        metrics = []
                        for key in ['total_pipeline_value', 'closed_won', 'win_rate_percent', 'total_weighted_forecast']:
                            if key in result:
                                metrics.append(f"{key}: {result[key]}")
                        if metrics:
                            st.markdown(" | ".join(metrics))
            
            st.divider()


def render_sidebar():
    """Render sidebar content."""
    with st.sidebar:
        st.title("📊 BI Agent")
        st.markdown("*Powered by Monday.com API + OpenAI*")
        
        st.divider()
        
        # Quick actions
        st.subheader("Quick Actions")
        
        if st.button("🔄 Refresh Boards List"):
            st.session_state.boards_cache = fetch_boards()
            st.success("Boards refreshed!")
        
        # Display boards
        if st.session_state.boards_cache:
            st.markdown("**Available Boards:**")
            for board in st.session_state.boards_cache:
                board_id = board.get('id', 'N/A')
                board_name = board.get('name', 'Unknown')
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**{board_name}**")
                        st.caption(f"ID: {board_id}")
                    with col2:
                        if st.button("📊", key=f"analyze_{board_id}"):
                            with st.spinner("Analyzing..."):
                                result = analyze_board(board_id)
                                if "error" not in result:
                                    st.session_state.last_analysis = result
                                    st.success("Analysis complete!")
                                    st.rerun()
                                else:
                                    st.error(result["error"])
        
        st.divider()
        
        # Settings
        st.subheader("Settings")
        st.session_state.show_trace = st.checkbox("Show Tool Traces", value=st.session_state.show_trace)
        
        if st.button("🗑️ Clear Conversation"):
            st.session_state.messages = []
            st.session_state.tool_trace = []
            st.rerun()
        
        st.divider()
        
        # API Status
        st.subheader("Status")
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                st.success("✅ API Connected")
            else:
                st.error("❌ API Error")
        except:
            st.error("❌ API Offline")


def render_chat_interface():
    """Render main chat interface."""
    st.markdown("## 💬 Ask Your BI Agent")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show tool trace for assistant messages
            if message["role"] == "assistant" and st.session_state.show_trace:
                if "tool_trace" in message and message["tool_trace"]:
                    render_tool_trace(message["tool_trace"])
    
    # Chat input
    if prompt := st.chat_input("Ask about your pipeline, revenue, conversions..."):
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing..."):
                result = send_query(prompt, st.session_state.session_id)
                
                if "error" in result and result["error"]:
                    st.error(f"Error: {result['error']}")
                    response_text = f"I encountered an error: {result['error']}"
                    tool_trace = []
                else:
                    response_text = result.get("response", "No response received")
                    tool_trace = result.get("tool_trace", [])
                    
                    # Display response
                    st.markdown(response_text)
                    
                    # Show trace
                    if st.session_state.show_trace and tool_trace:
                        render_tool_trace(tool_trace)
        
        # Add assistant message to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "tool_trace": tool_trace,
            "timestamp": datetime.now().isoformat()
        })
        
        # Rerun to update UI
        st.rerun()


def render_analysis_dashboard():
    """Render analysis dashboard if available."""
    if "last_analysis" in st.session_state:
        analysis = st.session_state.last_analysis
        
        st.divider()
        st.markdown("## 📈 Last Analysis")
        
        board_name = analysis.get("board_name", "Unknown Board")
        board_id = analysis.get("board_id", "N/A")
        
        st.markdown(f"**Board:** {board_name} (ID: {board_id})")
        
        results = analysis.get("results", {})
        
        # Create columns for metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Pipeline metrics
        if "pipeline" in results:
            pipeline = results["pipeline"]
            with col1:
                st.metric(
                    "Pipeline Value",
                    f"${pipeline.get('total_pipeline_value', 0):,.0f}"
                )
            with col2:
                st.metric(
                    "Deal Count",
                    f"{pipeline.get('deal_count', 0)}"
                )
        
        # Closed revenue
        if "closed_revenue" in results:
            closed = results["closed_revenue"]
            won = closed.get("closed_won", {})
            with col3:
                st.metric(
                    "Closed Won",
                    f"${won.get('revenue', 0):,.0f}"
                )
            with col4:
                st.metric(
                    "Win Rate",
                    f"{closed.get('win_rate_percent', 0):.1f}%"
                )
        
        # Detailed results
        with st.expander("📋 Detailed Results"):
            st.json(results)


def main():
    """Main application entry point."""
    init_session_state()
    
    # Initial boards fetch
    if not st.session_state.boards_cache:
        st.session_state.boards_cache = fetch_boards()
    
    # Render UI
    render_sidebar()
    
    # Main content area
    st.markdown("# 🤖 BI Agent for Monday.com")
    st.markdown("*Ask questions about your sales pipeline, revenue, conversions, and more.*")
    
    st.divider()
    
    # Example queries
    with st.expander("💡 Example Questions", expanded=False):
        examples = [
            "What's my total pipeline value?",
            "Show me conversion rates by stage",
            "What's my revenue by sector?",
            "Analyze board 5026873403",
            "Show quarterly trends for last year",
            "What's my win rate this quarter?",
            "Compare pipeline across all boards",
            "Show top 5 deals by value"
        ]
        
        cols = st.columns(2)
        for i, example in enumerate(examples):
            with cols[i % 2]:
                if st.button(example, key=f"example_{i}", use_container_width=True):
                    st.session_state.messages.append({
                        "role": "user",
                        "content": example,
                        "timestamp": datetime.now().isoformat()
                    })
                    st.rerun()
    
    st.divider()
    
    # Chat interface
    render_chat_interface()
    
    # Analysis dashboard
    render_analysis_dashboard()


if __name__ == "__main__":
    main()
