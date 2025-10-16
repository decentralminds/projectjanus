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
# Get the OpenRouter API key from the Agentverse secrets
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "YOUR_OPENROUTER_API_KEY_HERE")
# You can change the model to any one supported by OpenRouter
# Find models here: https://openrouter.ai/docs#models
MODEL = "openai/gpt-oss-20b:free"

# --- Agent and Protocol Setup ---
agent = Agent(
    name="janus_agent",
    seed="janus_agent_secret_seed_phrase_for_hackathon"
)
fund_agent_if_low(agent.wallet.address())

# Create a new protocol that is compatible with the chat protocol spec
protocol = Protocol(spec=chat_protocol_spec)


# --- Core Logic: Calling the LLM ---
def call_openrouter(prompt: str) -> str:
    """Sends a prompt to the OpenRouter API and returns the LLM's response."""
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY_HERE":
        return "Error: OpenRouter API key is not configured in Agentverse secrets."

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}"
            },
            data=json.dumps({
                "model": MODEL,
                "messages": [
                    # You can add a system prompt here if you want to define a persona
                    # {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
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
        return f"An unexpected error occurred: {e}"


# --- Message Handlers ---
@protocol.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    """Handles incoming chat messages from the user."""
    # Send an acknowledgement that we've received the message
    await ctx.send(
        sender,
        ChatAcknowledgement(timestamp=datetime.now(), acknowledged_msg_id=msg.msg_id),
    )

    # Extract the text content from the message
    text = ''.join(item.text for item in msg.content if isinstance(item, TextContent))
    ctx.logger.info(f"Received message from {sender}: {text}")

    # Query the model via OpenRouter
    response_text = call_openrouter(text)

    # Send the response back to the user
    await ctx.send(sender, ChatMessage(
        timestamp=datetime.utcnow(),
        msg_id=uuid4(),
        content=[
            TextContent(type="text", text=response_text),
            # Signal that the session is over. This tells the UI we aren't maintaining a long conversation history.
            EndSessionContent(type="end-session"),
        ]
    ))


@protocol.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    """Handles acknowledgement messages (e.g., for read receipts). We don't need to do anything here for now."""
    ctx.logger.info(f"Received acknowledgement from {sender} for message {msg.acknowledged_msg_id}")


# Attach the protocol to the agent and publish its manifest
agent.include(protocol, publish_manifest=True)

if __name__ == "__main__":
    agent.run()
