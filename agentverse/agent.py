import os
import requests
import json
import time  # Import the time module for delays
import random # Import the random module for jitter
from datetime import datetime
from uuid import uuid4
from typing import Tuple, Dict, Any, Optional

from uagents import Agent, Context, Protocol
from uagents.setup import fund_agent_if_low
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    EndSessionContent,
    TextContent,
    chat_protocol_spec,
)

# --- Configuration ---
# Get API keys from Agentverse secrets
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "YOUR_OPENROUTER_API_KEY_HERE")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "YOUR_TAVILY_API_KEY_HERE")

# Models and Endpoints
OPENROUTER_MODEL = "openai/gpt-oss-20b:free"
TAVILY_API_URL = "https://api.tavily.com/search"
BLOCKSCOUT_MCP_URL = "https://mcp.blockscout.com/mcp"
MAX_RETRIES = 1 
MAX_REASONING_STEPS = 7 # Max number of tool calls per query

# --- System Prompt ---
SYSTEM_PROMPT = """You are Janus, a world-class crypto analyst agent. Your goal is to provide accurate, up-to-date answers to user queries by combining web search with on-chain data analysis.

You have access to powerful tools. You can call them sequentially to gather information.

- **tavily_search(query: str):** Use this for real-time web information, news, or to find entities like wallet addresses.
- **blockscout_get_address_info(chain_id: str, address: str):** Use this to get a detailed profile of a specific crypto address.
- **blockscout_get_token_transfers_by_address(chain_id: str, address: str):** Use this to get the latest ERC-20 token transfers for an address. This is crucial for spotting recent purchases or sales.
- **blockscout_get_transactions_by_address(chain_id: str, address: str):** Use this to get the latest native coin (e.g., ETH) transactions for an address.

To use a tool, you must respond ONLY with a JSON object in the following format:
{"tool_name": "<tool_name>", "parameters": {"<param_name>": "<param_value>"}}

After using a tool, I will provide you with the results. You can then use another tool or, if you have enough information, provide a final comprehensive answer in plain text.
"""

# --- Agent and Protocol Setup ---
agent = Agent(
    name="janus_agent",
    seed="janus_agent_secret_seed_phrase_for_hackathon"
)
fund_agent_if_low(agent.wallet.address())

protocol = Protocol(spec=chat_protocol_spec)


# --- Core Logic Functions ---

def call_tavily_search(query: str, ctx: Context) -> str:
    """Performs a web search using the Tavily AI API."""
    if not TAVILY_API_KEY or TAVILY_API_KEY == "YOUR_TAVILY_API_KEY_HERE":
        return "Error: Tavily API key is not configured."
    try:
        response = requests.post(TAVILY_API_URL, json={"api_key": TAVILY_API_KEY, "query": query, "search_depth": "basic", "max_results": 5}, timeout=30)
        response.raise_for_status()
        return json.dumps(response.json().get("results", []))
    except requests.exceptions.RequestException as e:
        return f"Error connecting to Tavily API: {e}"

def call_blockscout_mcp(tool_name: str, arguments: dict, ctx: Context) -> str:
    """Calls a method on the Blockscout MCP API with the correct nested structure."""
    try:
        payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": tool_name, "arguments": arguments}}
        ctx.logger.info(f"Constructed Blockscout payload: {json.dumps(payload)}")
        headers = {'Accept': 'application/json, text/event-stream'}
        response = requests.post(BLOCKSCOUT_MCP_URL, json=payload, headers=headers, timeout=30, stream=True)
        response.raise_for_status()
        
        full_response_text = "".join(chunk.decode('utf-8') for chunk in response.iter_content(chunk_size=8192) if chunk)
        ctx.logger.info(f"Blockscout raw stream: {full_response_text}")

        if not full_response_text: raise ValueError("Empty response from Blockscout.")

        events = full_response_text.strip().split('event: message')
        last_event = next((event for event in reversed(events) if event.strip()), None)

        if last_event and 'data: ' in last_event:
            data_part = last_event.split('data: ', 1)[1]
            parsed_response = json.loads(data_part)

            if "error" in parsed_response:
                ctx.logger.error(f"Blockscout server error: {parsed_response['error']}")
                return json.dumps(parsed_response['error'])

            content = parsed_response.get("result", {}).get("content", [])
            if content and "text" in content[0]:
                # If the content is a nested JSON string, parse it.
                try:
                    nested_data = json.loads(content[0]["text"])
                    return json.dumps(nested_data)
                except json.JSONDecodeError:
                    # Otherwise, return the text directly.
                    return content[0]["text"]
            return json.dumps(parsed_response.get("result", {}))
        else:
            raise ValueError("No valid final data message in event stream.")
    except Exception as e:
        ctx.logger.error(f"Blockscout call failed: {e}")
        return f"Error during Blockscout call: {e}"

def call_openrouter(messages: list, ctx: Context) -> str:
    """Sends a list of messages to the OpenRouter API."""
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY_HERE":
        return "Error: OpenRouter API key is not configured."
    try:
        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "HTTP-Referer": "https://fetch.ai/agentverse", "X-Title": "Project Janus"}
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json={"model": OPENROUTER_MODEL, "messages": messages}, timeout=30)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        ctx.logger.error(f"OpenRouter call failed: {e}. Body: {e.response.text if e.response else 'N/A'}")
        return f"Error connecting to OpenRouter API: {e}"

# --- Helper Functions for Agentic Logic ---

def parse_tool_call(llm_output: str, ctx: Context) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Parses the LLM's output to find a valid tool call JSON."""
    try:
        start_token, end_token = "<|message|>", "<|call|>"
        start_index = llm_output.rfind(start_token)
        end_index = llm_output.rfind(end_token)

        if start_index != -1 and start_index < end_index:
            json_str = llm_output[start_index + len(start_token):end_index].strip()
            ctx.logger.info(f"Extracted JSON from special tokens: {json_str}")
            data = json.loads(json_str)
            
            if "tool_name" in data and "parameters" in data: 
                return data["tool_name"], data["parameters"]
            
            for key, value in data.items():
                if isinstance(value, dict):
                    if key in ["tavily_search", "blockscout_get_address_info", "blockscout_get_token_transfers_by_address", "blockscout_get_transactions_by_address"]:
                        return key, value
            
            ctx.logger.warning("Tool name not in JSON, attempting inference.")
            known_tools = ["blockscout_get_address_info", "blockscout_get_token_transfers_by_address", "blockscout_get_transactions_by_address", "tavily_search"]
            for tool in known_tools:
                if tool in llm_output:
                    ctx.logger.info(f"Inferred tool name: {tool}")
                    return tool, data
    except Exception as e:
        ctx.logger.warning(f"JSON parsing failed: {e}")
    return None, None

def execute_tool(tool_name: str, parameters: dict, ctx: Context) -> str:
    """Executes the appropriate tool by mapping the internal name to the API-specific name."""
    ctx.logger.info(f"Executing tool '{tool_name}' with params: {parameters}")

    # **FIX**: This map now correctly uses the FULL tool names required by the Blockscout API.
    tool_map = {
        "tavily_search": lambda p: call_tavily_search(p.get("query"), ctx),
        "blockscout_get_address_info": lambda p: call_blockscout_mcp("get_address_info", {"chain_id": str(p.get("chain_id")), "address": p.get("address")}, ctx),
        "blockscout_get_token_transfers_by_address": lambda p: call_blockscout_mcp("get_token_transfers_by_address", {"chain_id": str(p.get("chain_id")), "address": p.get("address")}, ctx),
        "blockscout_get_transactions_by_address": lambda p: call_blockscout_mcp("get_transactions_by_address", {"chain_id": str(p.get("chain_id")), "address": p.get("address")}, ctx),
    }
    
    if tool_name in tool_map:
        return tool_map[tool_name](parameters)
    
    ctx.logger.error(f"Unknown tool name: '{tool_name}'")
    return "Error: Unknown tool."

# --- Message Handlers ---
@protocol.on_message(model=ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    await ctx.send(sender, ChatAcknowledgement(timestamp=datetime.now(), acknowledged_msg_id=msg.msg_id))
    user_query = ''.join(item.text for item in msg.content if isinstance(item, TextContent))
    ctx.logger.info(f"Received query from {sender}: {user_query}")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_query}]
    final_response = ""

    for i in range(MAX_REASONING_STEPS):
        ctx.logger.info(f"Reasoning Step {i+1}/{MAX_REASONING_STEPS}")
        llm_decision_str = call_openrouter(messages, ctx)
        ctx.logger.info(f"LLM decision: {llm_decision_str}")

        tool_name, parameters = parse_tool_call(llm_decision_str, ctx)

        if tool_name and parameters:
            messages.append({"role": "assistant", "content": llm_decision_str})
            tool_results = execute_tool(tool_name, parameters, ctx)
            ctx.logger.info(f"Tool results: {tool_results}")
            messages.append({"role": "assistant", "content": f"I have retrieved the following data: {tool_results}. I will now decide the next step."})
        else:
            final_response = llm_decision_str
            break
    else:
        final_response = "I seem to be stuck in a reasoning loop. Could you please rephrase your question?"

    await ctx.send(sender, ChatMessage(timestamp=datetime.utcnow(), msg_id=uuid4(), content=[TextContent(type="text", text=final_response), EndSessionContent(type="end-session")]))

@protocol.on_message(model=ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    ctx.logger.info(f"Received ack from {sender} for msg {msg.acknowledged_msg_id}")

agent.include(protocol, publish_manifest=True)

if __name__ == "__main__":
    agent.run()

