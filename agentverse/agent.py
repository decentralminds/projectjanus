import os
import requests
import json
import time  # Import the time module for delays
import random # Import the random module for jitter
from datetime import datetime
from uuid import uuid4

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

# This is the main system prompt that gives the LLM its instructions and tools.
SYSTEM_PROMPT = """You are Janus, a world-class crypto analyst agent. Your goal is to provide accurate, up-to-date answers to user queries by combining web search with on-chain data analysis.

You have access to powerful tools to help you:
- **tavily_search(query: str):** Use this for real-time web information, news, or market sentiment.
- **blockscout_get_address_info(chain_id: str, address: str):** Use this to get detailed on-chain data for a specific crypto address. The chain_id for Ethereum Mainnet is "1".

To use a tool, you must respond ONLY with a JSON object in the following format:
{"tool_name": "<tool_name>", "parameters": {"<param_name>": "<param_value>"}}

When you receive the results from a tool, you must use them to formulate a comprehensive, human-readable answer to the user's original question.
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
    """Performs a web search using the Tavily AI API with retry logic."""
    if not TAVILY_API_KEY or TAVILY_API_KEY == "YOUR_TAVILY_API_KEY_HERE":
        return "Error: Tavily API key is not configured in Agentverse secrets."
    
    for attempt in range(MAX_RETRIES):
        try:
            ctx.logger.info(f"Performing Tavily search for: {query} (Attempt {attempt + 1})")
            response = requests.post(TAVILY_API_URL, json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "search_depth": "basic",
                "include_answer": True,
                "max_results": 5
            }, timeout=30)
            response.raise_for_status()
            results = response.json()
            return json.dumps(results.get("results", []))

        except requests.exceptions.RequestException as e:
            ctx.logger.error(f"Tavily API call failed on attempt {attempt + 1}: {e}")
            if attempt < MAX_RETRIES - 1:
                wait_time = (2 ** (attempt + 1)) + (random.random() * 2)
                ctx.logger.info(f"Waiting for {wait_time:.2f} seconds before retrying...")
                time.sleep(wait_time)
            else:
                return f"Error connecting to Tavily API after {MAX_RETRIES} attempts: {e}"
        except Exception as e:
            return f"An unexpected error occurred during search: {e}"
    return "Error: Tavily search failed after all retries."


def call_blockscout_mcp(tool_name: str, arguments: dict, ctx: Context) -> str:
    """Calls a method on the Blockscout MCP API with the correct nested structure."""
    for attempt in range(MAX_RETRIES):
        last_event = "" 
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }
            ctx.logger.info(f"Constructed Blockscout payload: {json.dumps(payload)}")
            
            headers = {
                'Accept': 'application/json, text/event-stream',
            }
            
            response = requests.post(BLOCKSCOUT_MCP_URL, json=payload, headers=headers, timeout=30, stream=True)
            response.raise_for_status()
            
            full_response_text = ""
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    full_response_text += chunk.decode('utf-8')
            
            ctx.logger.info(f"Blockscout raw response stream: {full_response_text}")

            if not full_response_text:
                raise ValueError("Received empty response from Blockscout server.")

            events = full_response_text.strip().split('event: message')
            last_event = next((event for event in reversed(events) if event.strip()), None)

            if last_event and 'data: ' in last_event:
                data_part = last_event.split('data: ')[1]
                parsed_response = json.loads(data_part)

                if "error" in parsed_response:
                    error_details = parsed_response["error"]
                    ctx.logger.error(f"Blockscout server returned an error: {error_details}")
                    return json.dumps(error_details) 

                content = parsed_response.get("result", {}).get("content", [])
                if content and isinstance(content, list) and "text" in content[0]:
                    nested_json_string = content[0]["text"]
                    final_data = json.loads(nested_json_string)
                    return json.dumps(final_data)
                else:
                    return json.dumps(parsed_response.get("result", {}))
            else:
                raise ValueError("Could not find a valid final data message in the event stream.")
        
        except (requests.exceptions.RequestException, ValueError) as e:
            ctx.logger.error(f"Blockscout API call failed on attempt {attempt + 1}: {e}")
            if attempt < MAX_RETRIES - 1:
                wait_time = (2 ** (attempt + 1)) + (random.random() * 2)
                ctx.logger.info(f"Waiting for {wait_time:.2f} seconds before retrying...")
                time.sleep(wait_time)
            else:
                return f"Error connecting to Blockscout API after {MAX_RETRIES} attempts: {e}"
        except json.JSONDecodeError as e:
            ctx.logger.error(f"Blockscout JSON decode failed: {e}. Offending text: '{last_event}'")
            return f"Error decoding Blockscout response: {e}"
        except Exception as e:
            return f"An unexpected error occurred during Blockscout call: {e}"
    return f"Error: Blockscout MCP call failed for tool '{tool_name}' after all retries."


def call_openrouter(messages: list, ctx: Context) -> str:
    """Sends a list of messages to the OpenRouter API and returns the LLM's response with retry logic."""
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY_HERE":
        return "Error: OpenRouter API key is not configured."

    for attempt in range(MAX_RETRIES):
        try:
            ctx.logger.info(f"Calling OpenRouter API (Attempt {attempt + 1})")
            
            payload = {
                "model": OPENROUTER_MODEL,
                "messages": messages
            }
            
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": "https://fetch.ai/agentverse",
                "X-Title": "Project Janus Hackathon"
            }
            
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()

            if result.get('choices') and result['choices'][0].get('message'):
                return result['choices'][0]['message']['content']
            else:
                return "An error occurred: The API response was not in the expected format."
        
        except requests.exceptions.RequestException as e:
            ctx.logger.error(f"OpenRouter API call failed on attempt {attempt + 1}: {e}")
            if e.response is not None:
                ctx.logger.error(f"Error response body: {e.response.text}")
            
            if attempt < MAX_RETRIES - 1:
                wait_time = (2 ** (attempt + 1)) + (random.random() * 2)
                ctx.logger.info(f"Waiting for {wait_time:.2f} seconds before retrying...")
                time.sleep(wait_time)
            else:
                 return f"Error connecting to OpenRouter API after {MAX_RETRIES} attempts: {e}"
        except Exception as e:
            return f"An unexpected error occurred with the LLM call: {e}"
    return "Error: OpenRouter call failed after all retries."


# --- Message Handlers ---
@protocol.on_message(model=ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    await ctx.send(
        sender,
        ChatAcknowledgement(timestamp=datetime.now(), acknowledged_msg_id=msg.msg_id),
    )

    user_query = ''.join(item.text for item in msg.content if isinstance(item, TextContent))
    ctx.logger.info(f"Received query from {sender}: {user_query}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]
    
    llm_decision_str = call_openrouter(messages, ctx)
    ctx.logger.info(f"LLM decision: {llm_decision_str}")

    final_response = ""
    try:
        tool_name_found = None
        if "blockscout_get_address_info" in llm_decision_str:
            tool_name_found = "blockscout_get_address_info"
        elif "tavily_search" in llm_decision_str:
            tool_name_found = "tavily_search"

        if tool_name_found:
            ctx.logger.info(f"Inferred tool from context: '{tool_name_found}'")

            start_brace = llm_decision_str.rfind('{')
            end_brace = llm_decision_str.rfind('}') + 1
            
            if start_brace != -1 and end_brace != 0 and start_brace < end_brace:
                json_str = llm_decision_str[start_brace:end_brace]
                decision_json = json.loads(json_str)
                parameters = decision_json.get("parameters", decision_json)
                
                tool_results = ""
                if tool_name_found == "tavily_search":
                    search_query = parameters.get("query")
                    if search_query:
                        tool_results = call_tavily_search(search_query, ctx)
                elif tool_name_found == "blockscout_get_address_info":
                    chain_id = parameters.get("chain_id")
                    address = parameters.get("address")
                    if chain_id and address:
                        tool_results = call_blockscout_mcp(
                            "get_address_info", 
                            {"chain_id": str(chain_id), "address": address}, 
                            ctx
                        )
                
                if tool_results:
                    try:
                        tool_data = json.loads(tool_results)
                        if 'data' in tool_data: 
                            pruned_data = {
                                "basic_info": tool_data.get("data", {}).get("basic_info"),
                                "tags": [tag.get("name") for tag in tool_data.get("data", {}).get("metadata", {}).get("tags", [])]
                            }
                            tool_results = json.dumps(pruned_data)
                            ctx.logger.info(f"Pruned Blockscout data for LLM: {tool_results}")
                    except (json.JSONDecodeError, TypeError):
                        ctx.logger.warning("Could not prune tool results, passing raw.")

                    ctx.logger.info("Sending tool results to LLM for final synthesis...")
                    
                    # **FIX**: Re-structured the synthesis message to be a simple assistant message
                    # This avoids the unsupported 'role: tool' and is more compatible.
                    synthesis_messages = messages + [
                        {
                            "role": "assistant",
                            "content": f"I have retrieved the following on-chain data: {tool_results}. Now I will formulate a response."
                        }
                    ]

                    ctx.logger.info(f"Final synthesis payload for OpenRouter: {json.dumps(synthesis_messages)}")
                    final_response = call_openrouter(synthesis_messages, ctx)
                else:
                    final_response = "I tried to use a tool, but I couldn't find the necessary parameters in your request. Please be more specific."
            else:
                final_response = "I tried to use a tool, but something went wrong with its format. Please try again."
        else:
            final_response = llm_decision_str

    except (json.JSONDecodeError, TypeError):
        final_response = llm_decision_str
    except NameError as e:
        ctx.logger.error(f"A NameError occurred: {e}")
        final_response = "I encountered an internal error. The developer has been notified."


    await ctx.send(sender, ChatMessage(
        timestamp=datetime.utcnow(),
        msg_id=uuid4(),
        content=[TextContent(type="text", text=final_response), EndSessionContent(type="end-session")]
    ))


@protocol.on_message(model=ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    ctx.logger.info(f"Received acknowledgement from {sender} for message {msg.acknowledged_msg_id}")

agent.include(protocol, publish_manifest=True)

if __name__ == "__main__":
    agent.run()

