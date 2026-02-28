"""ReAct-style BI Agent with Groq API Function Calling."""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from groq import Groq

from app.tools import get_tool_registry, ToolRegistry
from app.memory import get_memory_manager, MemoryManager

logger = logging.getLogger(__name__)


class BIAgent:
    """ReAct-style Business Intelligence Agent using Groq function calling."""
    
    SYSTEM_PROMPT = """You are a Business Intelligence Agent specialized in analyzing sales and pipeline data from Monday.com.

Your core principles:
1. ALWAYS fetch live data using the available tools - never assume or make up data
2. Think step-by-step: understand the question, fetch data, analyze, provide insights
3. Show your reasoning process transparently
4. Handle messy data gracefully through the cleaning layer
5. Provide actionable business insights, not just raw numbers

When answering:
- Start by understanding what data is needed
- Call the appropriate fetch tools first
- Use analysis tools to compute metrics
- Interpret the results in business context
- Highlight key insights and trends
- Suggest actions when relevant

You have access to these tool categories:
- Discovery: list_boards, fetch_board_data, fetch_multiple_boards
- Pipeline Analysis: analyze_pipeline, analyze_conversions
- Revenue Analysis: get_closed_revenue, compute_forecast
- Segmentation: analyze_sector_breakdown
- Time-based: analyze_quarterly_trends (with time filters)
- Cross-board: cross_board_analysis

CRITICAL: Always fetch data before analyzing. Never use cached or assumed values."""
    
    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.1-8b-instant"  # Fast, cost-effective model on Groq
        self.tools = get_tool_registry()
        self.memory = get_memory_manager()
        self.conversation_history: List[Dict] = []
        self.current_trace: List[Dict] = []
        logger.info("BIAgent initialized with Groq")
    
    def _create_messages(self, user_query: str) -> List[Dict]:
        """Create message payload for Groq API."""
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT}
        ]
        
        # Add conversation history (last 5 exchanges)
        recent_history = self.conversation_history[-10:]
        messages.extend(recent_history)
        
        # Add user query with context from memory if available
        context = self._get_relevant_context(user_query)
        if context:
            enhanced_query = f"User question: {user_query}\n\nRelevant context from previous analysis:\n{context}"
            messages.append({"role": "user", "content": enhanced_query})
        else:
            messages.append({"role": "user", "content": user_query})
        
        return messages
    
    def _get_relevant_context(self, query: str) -> str:
        """Get relevant context from memory."""
        # Check if we have recent board analyses
        recent_analyses = self.memory.get_recent_analyses(limit=3)
        if recent_analyses:
            context_parts = []
            for analysis in recent_analyses:
                context_parts.append(
                    f"- Previously analyzed board '{analysis.get('board_name', 'Unknown')}' "
                    f"(ID: {analysis.get('board_id', 'N/A')})"
                )
            return "\n".join(context_parts)
        return ""
    
    def _call_llm(self, messages: List[Dict], tools: Optional[List[Dict]] = None) -> Any:
        """Call Groq API with messages and optional tools."""
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.2,  # Low temperature for more deterministic responses
            }
            
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            
            response = self.client.chat.completions.create(**kwargs)
            return response
            
        except Exception as e:
            logger.error(f"Groq API error: {str(e)}")
            raise
    
    def _execute_tool_calls(self, tool_calls: List[Any]) -> List[Dict]:
        """Execute tool calls and return results."""
        results = []
        
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            try:
                params = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                params = {}
            
            logger.info(f"Executing tool: {tool_name} with params: {params}")
            
            # Execute the tool
            result = self.tools.execute_tool(tool_name, params)
            
            # Store in trace
            trace_entry = {
                "timestamp": datetime.now().isoformat(),
                "tool": tool_name,
                "parameters": params,
                "result": result
            }
            self.current_trace.append(trace_entry)
            
            # Store in memory if it's a board analysis
            if tool_name in ["analyze_pipeline", "analyze_conversions", "analyze_sector_breakdown",
                           "analyze_quarterly_trends", "compute_forecast", "get_closed_revenue"]:
                if "board_id" in result and "board_name" in result:
                    self.memory.store_analysis({
                        "board_id": result["board_id"],
                        "board_name": result["board_name"],
                        "analysis_type": tool_name,
                        "result": result,
                        "timestamp": datetime.now().isoformat()
                    })
            
            results.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": tool_name,
                "content": json.dumps(result)
            })
        
        return results
    
    def process_query(self, user_query: str) -> Dict:
        """Process a user query through the ReAct loop."""
        logger.info(f"Processing query: {user_query}")
        
        # Reset trace for new query
        self.current_trace = []
        
        # Step 1: Create messages with history
        messages = self._create_messages(user_query)
        
        # Step 2: Get tool definitions
        tool_definitions = self.tools.get_tool_definitions()
        
        max_iterations = 5
        iteration = 0
        last_tool_result = None
        
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"ReAct iteration {iteration}")
            
            # Call LLM
            response = self._call_llm(messages, tool_definitions)
            message = response.choices[0].message
            
            # Add assistant message to conversation
            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [tc.model_dump() for tc in message.tool_calls] if message.tool_calls else []
            })
            
            # Check if there are tool calls
            if message.tool_calls:
                # Execute tools
                tool_results = self._execute_tool_calls(message.tool_calls)
                
                # Store the last tool result for potential final answer
                if tool_results:
                    last_tool_result = tool_results[-1]["content"]
                
                # Add tool results to messages
                for result in tool_results:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": result["tool_call_id"],
                        "name": result["name"],
                        "content": result["content"]
                    })
                
                # Continue loop to let LLM process results
                continue
            
            else:
                # No tool calls - we have the final response
                final_response = message.content
                
                # Update conversation history
                self.conversation_history.append({
                    "role": "user",
                    "content": user_query
                })
                self.conversation_history.append({
                    "role": "assistant",
                    "content": final_response
                })
                
                # Trim history if too long
                if len(self.conversation_history) > 20:
                    self.conversation_history = self.conversation_history[-20:]
                
                return {
                    "response": final_response,
                    "tool_trace": self.current_trace,
                    "iterations": iteration
                }
        
        # Max iterations reached - return last tool result as final answer
        if last_tool_result:
            # Try to parse and format the last tool result
            try:
                result_data = json.loads(last_tool_result)
                # Format a business-friendly response from the tool result
                if isinstance(result_data, dict):
                    if "board_name" in result_data:
                        formatted_response = f"**Analysis for {result_data['board_name']}**\n\n"
                        
                        # Add key metrics based on what was computed
                        if "total_pipeline_value" in result_data:
                            formatted_response += f"**Total Pipeline Value:** ${result_data['total_pipeline_value']:,.2f}\n"
                        if "weighted_pipeline_value" in result_data:
                            formatted_response += f"**Weighted Pipeline Value:** ${result_data['weighted_pipeline_value']:,.2f}\n"
                        if "total_closed_revenue" in result_data:
                            formatted_response += f"**Closed Revenue:** ${result_data['total_closed_revenue']:,.2f}\n"
                        if "conversion_rate" in result_data:
                            formatted_response += f"**Conversion Rate:** {result_data['conversion_rate']:.1%}\n"
                        if "item_count" in result_data:
                            formatted_response += f"**Total Items:** {result_data['item_count']}\n"
                        
                        # Add any insights if present
                        if "insights" in result_data and result_data["insights"]:
                            formatted_response += f"\n**Key Insights:**\n"
                            for insight in result_data["insights"]:
                                formatted_response += f"- {insight}\n"
                        
                        # Add raw data summary
                        formatted_response += f"\n*Raw result data available in tool trace*"
                        
                        final_response = formatted_response
                    else:
                        final_response = f"**Analysis Complete**\n\n```json\n{json.dumps(result_data, indent=2)}\n```"
                else:
                    final_response = str(result_data)
                    
            except json.JSONDecodeError:
                final_response = f"**Analysis Result:**\n\n{last_tool_result}"
            
            # Update conversation history
            self.conversation_history.append({
                "role": "user",
                "content": user_query
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": final_response
            })
            
            # Trim history if too long
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
            
            return {
                "response": final_response,
                "tool_trace": self.current_trace,
                "iterations": iteration
            }
        
        # No tool results available - return graceful message
        return {
            "response": "I was unable to complete the analysis. Please try a more specific query about your Monday.com data.",
            "tool_trace": self.current_trace,
            "iterations": iteration
        }
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation."""
        exchanges = len(self.conversation_history) // 2
        return f"Conversation contains {exchanges} exchanges"
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        self.current_trace = []
        logger.info("Conversation history cleared")


# Convenience function for direct usage
def ask_agent(question: str, api_key: Optional[str] = None) -> Dict:
    """Ask the BI Agent a question and get response with tool traces."""
    agent = BIAgent(api_key=api_key)
    return agent.process_query(question)
