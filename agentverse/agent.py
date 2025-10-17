import os
import requests
import json
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

# This is the new system prompt that gives the LLM its instructions and tools
SYSTEM_PROMPT = """You are Janus, a world-class crypto analyst agent. Your goal is to provide accurate, up-to-date answers to user queries.

You have access to a powerful tool to help you:
- **tavily_search(query: str):** Use this tool ONLY when you need to find real-time information, breaking news, or specific data points from the web to answer the user's question. Do not use it for general knowledge questions.

To use the tool, you must respond ONLY with a JSON object in the following format:
{"tool_name": "tavily_search", "parameters": {"query": "<your search query here>"}}

If you can answer the question directly without needing a web search, provide the answer in plain text.
"""

# --- Agent and Protocol Setup ---
agent = Agent(
    name="janus_agent",
    seed="janus_agent_secret_seed_phrase_for_hackathon"
)
fund_agent_if_low(agent.wallet.address())

protocol = Protocol(spec=chat_protocol_spec)


# --- Core Logic Functions ---

def call_tavily_search(query: str) -> str:
    """Performs a web search using the Tavily AI API."""
    if not TAVILY_API_KEY or TAVILY_API_KEY == "YOUR_TAVILY_API_KEY_HERE":
        return "Error: Tavily API key is not configured in Agentverse secrets."
    ctx.logger.info(f"Performing Tavily search for: {query}")
    try:
        response = requests.post(TAVILY_API_URL, json={
            "api_key": TAVILY_API_KEY,
            "query": query,
            "search_depth": "basic",
            "include_answer": True,
            "max_results": 5
        })
        response.raise_for_status()
        results = response.json()
        
        # We'll just return the raw search results for the LLM to process
        return json.dumps(results.get("results", []))

    except requests.exceptions.RequestException as e:
        return f"Error connecting to Tavily API: {e}"
    except Exception as e:
        return f"An unexpected error occurred during search: {e}"


def call_openrouter(messages: list) -> str:
    """Sends a list of messages to the OpenRouter API and returns the LLM's response."""
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY_HERE":
        return "Error: OpenRouter API key is not configured."

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            data=json.dumps({
                "model": OPENROUTER_MODEL,
                "messages": messages
            })
        )
        response.raise_for_status()
        result = response.json()

        if result.get('choices') and result['choices'][0].get('message'):
            return result['choices'][0]['message']['content']
        else:
            return "An error occurred: The API response was not in the expected format."

    except requests.exceptions.RequestException as e:
        return f"Error connecting to OpenRouter API: {e}"
    except Exception as e:
        return f"An unexpected error occurred with the LLM call: {e}"


# --- Message Handlers ---
@protocol.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    """Handles incoming chat messages and orchestrates the agent's response."""
    await ctx.send(
        sender,
        ChatAcknowledgement(timestamp=datetime.now(), acknowledged_msg_id=msg.msg_id),
    )

    user_query = ''.join(item.text for item in msg.content if isinstance(item, TextContent))
    ctx.logger.info(f"Received query from {sender}: {user_query}")

    # Step 1: First LLM call to decide on a plan (use a tool or answer directly)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]
    
    ctx.logger.info("Asking LLM for initial plan...")
    llm_decision_str = call_openrouter(messages)
    ctx.logger.info(f"LLM decision: {llm_decision_str}")

    final_response = ""
    try:
        # Check if the LLM responded with a JSON tool call
        decision_json = json.loads(llm_decision_str)
        if decision_json.get("tool_name") == "tavily_search":
            search_query = decision_json["parameters"]["query"]
            ctx.logger.info(f"LLM requested tool: tavily_search with query: '{search_query}'")

            # Step 2: Execute the tool call
            search_results = call_tavily_search(search_query)
            
            # Step 3: Second LLM call to synthesize a final answer from the search results
            ctx.logger.info("Sending search results to LLM for final synthesis...")
            synthesis_messages = messages + [
                {"role": "assistant", "content": llm_decision_str}, # The tool request
                {"role": "tool", "content": search_results}       # The tool's output
            ]
            final_response = call_openrouter(synthesis_messages)
        else:
            # The JSON wasn't a valid tool call, treat as a direct answer
            final_response = llm_decision_str

    except (json.JSONDecodeError, TypeError):
        # The response was not JSON, so it's a direct answer from the first call
        final_response = llm_decision_str

    await ctx.send(sender, ChatMessage(
        timestamp=datetime.utcnow(),
        msg_id=uuid4(),
        content=[
            TextContent(type="text", text=final_response),
            EndSessionContent(type="end-session"),
        ]
    ))


@protocol.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    ctx.logger.info(f"Received acknowledgement from {sender} for message {msg.acknowledged_msg_id}")


agent.include(protocol, publish_manifest=True)

if __name__ == "__main__":
    agent.run()

