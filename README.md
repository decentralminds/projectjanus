# **Project Janus \- Your On-Chain Oracle Agent**

**Project Janus** is an intelligent AI agent designed to provide insightful analysis of the crypto landscape by seamlessly integrating real-time on-chain data with off-chain web information. Built for the Ethereum Hackathon, Janus acts as a knowledgeable assistant, helping users understand market movements, track key players, and monitor network health.

## **Core Idea**

In the fast-paced world of cryptocurrency, information is fragmented. On-chain data provides ground truth but lacks context, while news and social media offer context but can be noisy or inaccurate. Project Janus bridges this gap by leveraging the power of AI to:

1. **Fetch** live blockchain data using the Blockscout Model Context Protocol (MCP).  
2. **Search** the web for relevant news and context using Tavily AI.  
3. **Reason** using a Large Language Model (LLM) via OpenRouter to synthesize information and answer user questions in a natural, conversational way.

Users interact with Janus through the intuitive ASI:one chat interface.

## **Tech Stack**

* **Agent Framework:** [Fetch.ai Agentverse](https://agentverse.ai/)  
* **User Interface:** [ASI:one](https://asi.one/)  
* **On-Chain Data:** Public [Blockscout MCP](https://mcp.blockscout.com/mcp) Endpoint  
* **Web Search:** [Tavily AI API](https://tavily.com/)  
* **LLM Provider:** [OpenRouter.ai](https://openrouter.ai/) (using various models like openai/gpt-oss-20b:free)  
* **Language:** Python  
* **Core Libraries:** uagents, requests

## **Implemented Capabilities**

As of now, Project Janus can:

1. **Synthesize On-chain & Off-chain Insights:** Answer complex questions like *"Why is \[Token Name\] price down today?"* by performing multi-step reasoning: searching the web for news and using Blockscout tools to check for relevant on-chain activity (e.g., large transfers).  
2. **Track Key Wallets:** Find the Ethereum address associated with prominent figures (e.g., *"Find Justin Sun's address"*) using web search, and then retrieve and summarize recent significant transactions (ETH or ERC-20 tokens) using Blockscout's get\_transactions\_by\_address and get\_token\_transfers\_by\_address tools.  
3. **Analyze Gas Prices:** Fetch the current Ethereum base gas fee using Blockscout's get\_latest\_block tool and, combined with web search, provide context on potential causes for high gas fees.

## **Future Enhancements**

Project Janus has a strong foundation for further development:

* **Opportunity Spotting on L2s (e.g., Base Chain):** Integrate more Blockscout tools (or chain-specific APIs if needed) to actively scan for potential opportunities like newly deployed contracts, tokens experiencing high volume surges, or notable wallet activities on Layer 2 networks.  
* **Deeper Gas Analysis:** Implement logic to analyze historical gas data or specific transaction details to provide more granular insights into gas spikes.  
* **Expanded Toolset:** Integrate additional Blockscout MCP tools (e.g., get\_contract\_abi, inspect\_contract\_code, lookup\_token\_by\_symbol) for deeper smart contract analysis.  
* **Proactive Alerts:** Develop capabilities for the agent to proactively monitor specific addresses or network conditions and alert the user via ASI:one.

## **Getting Started (Conceptual)**

*(This section would typically include setup instructions if it were a standalone project)*

1. Deploy the agent code to Fetch.ai Agentverse.  
2. Add OPENROUTER\_API\_KEY and TAVILY\_API\_KEY as secrets in Agentverse options.  
3. Run the agent.  
4. Add the agent's address to the ASI:one interface.  
5. Start chatting\!

*Project Janus \- Bringing clarity to the blockchain.*