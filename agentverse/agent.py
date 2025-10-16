import os
import requests
import json
from uagents import Agent, Context, Model
from uagents.setup import fund_agent_if_low

# Get the OpenRouter API key from the Agentverse secrets
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "YOUR_OPENROUTER_API_KEY_HERE")

# Define the message model that the agent will receive
class Message(Model):
    text: str

# Create the agent
agent = Agent(
    name="janus_agent",
    port=8000,
    seed="janus_agent_seed_phrase_for_hackathon",
    endpoint=["http://127.0.0.1:8000/submit"],
)

# Fund the agent if its balance is low
fund_agent_if_low(agent.wallet.address())

# Function to call the OpenRouter API
def call_openrouter(prompt: str) -> str:
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY_HERE":
        return "Error: OpenRouter API key not set."

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}"
            },
            data=json.dumps({
                "model": "openai/gpt-oss-20b:free", # Using a good free model to start
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            })
        )
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        result = response.json()
        if result['choices']:
            return result['choices'][0]['message']['content']
        else:
            return "I'm sorry, I couldn't get a response. Please try again."
    except requests.exceptions.RequestException as e:
        print(f"API Call Error: {e}")
        return f"Error connecting to OpenRouter: {e}"
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return "An unexpected error occurred."


# Register an event handler for when the agent receives a message
@agent.on_message(model=Message)
async def handle_message(ctx: Context, sender: str, msg: Message):
    ctx.logger.info(f"Received message from {sender}: {msg.text}")

    # Get the LLM's response
    llm_response = call_openrouter(msg.text)

    # Send the response back to the user
    await ctx.send(sender, Message(text=llm_response))

if __name__ == "__main__":
    agent.run()