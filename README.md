# AI BI Agent Assessment

A **ReAct-style Business Intelligence Agent** with live Monday.com API integration. This system fetches real-time data, handles messy data normalization, performs cross-board analytics, and provides actionable business insights through a conversational interface.

## Architecture Overview

```
User (Founder Question)
        ↓
Frontend (Streamlit Chat UI)
        ↓
LLM Agent (ReAct Reasoning Layer with OpenAI Function Calling)
        ↓
Tool Calling Layer
        ↓
Monday.com API Client (Live Data Fetch, No Caching)
        ↓
Data Cleaning + Normalization Layer
        ↓
BI Analysis Engine (Pipeline, Conversions, Sector Breakdown, Forecasting)
        ↓
Insight Generator + Response with Tool Trace
```

## Features

- **Live API Integration**: Fetches real-time data from Monday.com at query time (no caching)
- **Data Normalization**: Handles messy data (nulls, inconsistent sectors, varied revenue formats, missing stages)
- **ReAct-Style Agent**: Uses OpenAI function calling with iterative reasoning
- **Full Tool Trace**: Shows every API call and analysis step
- **Cross-Board Analysis**: Can analyze and compare multiple boards
- **BI Analytics**: Pipeline value, conversion rates, sector breakdowns, quarterly trends, forecasting
- **Conversational Memory**: Maintains context across queries
- **Time Filtering**: Analyze specific time periods (this quarter, last month, YTD, etc.)

## Project Structure

```
app/
├── monday_client.py    # Monday.com API client with full logging, pagination
├── data_clean.py       # Data normalization layer (fuzzy matching, date parsing, revenue standardization)
├── analytics.py        # BI engine (pipeline, conversions, sector breakdowns, forecasting)
├── tools.py            # Tool registry for agent (OpenAI function definitions)
├── agent.py            # ReAct-style LLM agent with OpenAI function calling
├── memory.py           # Conversational memory system
├── main.py             # FastAPI backend
├── frontend.py         # Streamlit chat UI
├── config.py           # Configuration settings
└── test_monday_api.py  # API test script

.env                    # Environment variables (API keys)
requirements.txt        # Python dependencies
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Edit `.env` file:

```
MONDAY_API_KEY=your_monday_api_key
OPENAI_API_KEY=your_openai_api_key
```

### 3. Run the Backend (FastAPI)

```bash
cd app
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### 4. Run the Frontend (Streamlit)

In a new terminal:

```bash
cd app
streamlit run frontend.py
```

The UI will open at `http://localhost:8501`

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check with Monday.com status |
| `/query` | POST | Process natural language query |
| `/boards` | GET | List available Monday.com boards |
| `/boards/{id}` | GET | Get specific board details |
| `/analyze` | POST | Run analysis on a board |
| `/analyze/cross-board` | POST | Analyze multiple boards |
| `/tools` | GET | List available tools |
| `/sessions/{id}/trace` | GET | Get session tool trace |
| `/sessions/{id}` | DELETE | Clear session |

## Example Queries

The agent can handle queries like:

- "What's my total pipeline value?"
- "Show me conversion rates by stage"
- "What's my revenue by sector?"
- "Analyze board 5026873403"
- "Show quarterly trends for last year"
- "What's my win rate this quarter?"
- "Compare pipeline across all boards"
- "Show top 5 deals by value"

## Data Cleaning Features

The `data_clean.py` module handles:

- **Null/Empty Values**: Drop or fill strategies
- **Revenue Normalization**: Parses formats like "$1.5M", "2.5k", "50000"
- **Sector Standardization**: Fuzzy matching against standard sectors (Tech, Healthcare, Finance, etc.)
- **Stage Normalization**: Maps various stage names to standard pipeline stages
- **Date Parsing**: Handles multiple date formats
- **Time Features**: Extracts year, quarter, month for trend analysis

## BI Analytics Capabilities

The `analytics.py` engine computes:

- **Pipeline Value**: Total pipeline, stage breakdown, average deal size
- **Closed Revenue**: Won vs lost revenue, win rates
- **Conversion Rates**: Stage-to-stage conversion analysis
- **Sector Breakdown**: Revenue and win rates by industry
- **Quarterly Trends**: QoQ growth, seasonal patterns
- **Forecasting**: Weighted revenue forecast based on stage probabilities

## Tool Trace

Every query shows a complete trace of:
1. Which tools were called
2. What parameters were used
3. What results were returned
4. How the LLM interpreted the results

This transparency ensures the agent is working with live data and shows its reasoning process.

## Configuration

Environment variables (in `.env`):

| Variable | Description | Default |
|----------|-------------|---------|
| `MONDAY_API_KEY` | Monday.com API token | Required |
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `OPENAI_MODEL` | LLM model to use | gpt-4o |
| `API_PORT` | Backend API port | 8000 |
| `LOG_LEVEL` | Logging level | INFO |

## Testing the Monday.com API

Run the test script to verify API connectivity:

```bash
cd app
python test_monday_api.py
```

## Development

### Running Tests

```bash
cd app
python -c "from monday_client import get_monday_client; c = get_monday_client(); print(c.get_boards(limit=3))"
```

### Adding New Tools

1. Add handler method in `tools.py`
2. Add tool definition in `get_tool_definitions()`
3. Register handler in `_register_all_tools()`

## License

MIT License

## Notes

- **No Caching**: The system fetches fresh data for every query as per requirements
- **Tool Traces**: Full visibility into every API call and analysis step
- **Live Data**: All analysis is performed on current Monday.com data
- **Ambiguous Queries**: The agent uses reasoning to interpret and clarify business questions