# **Project Canvas: Project Janus**

Status: Not Started  
Timeline: 2 Weeks (Approx. 1 hour/day \+ weekends)

## **🎯 Core Idea**

A chat-based AI agent, deployed on **Fetch.ai Agentverse** and accessed via **ASI:one**, that synthesizes on-chain data from **Blockscout** with real-world news from the **Tavily AI** search API. The agent will help users understand crypto market movements, identify on-chain anomalies, and potentially spot new opportunities.

## **🛠️ Technology Stack**

| Component | Service / Tool | Purpose |
| :---- | :---- | :---- |
| **Agent Hosting** | Fetch.ai Agentverse | To run the core Python logic of our agent. |
| **User Interface** | ASI:one | The chat interface to interact with the agent. |
| **On-Chain Data** | Public Blockscout MCP | For all blockchain data (gas, transactions, tokens). |
| **Web Search/News** | Tavily AI API | To fetch relevant news and context for market events. |
| **Core Logic** | Python (requests) | The programming language for the agent. |

## **Kanban Project Board**

*Move tasks across columns to track progress.*

### **📋 To-Do (Backlog)**

| ID | Task | Details |
| :---- | :---- | :---- |
| **1** | Deploy Basic Agentverse Agent | Get a "Hello World" agent running on Agentverse and connected to ASI:one. |
| **2** | Get Tavily AI API Key | Sign up for Tavily AI and secure the API key. |
| **3** | Connect to Blockscout API | Make a test call from the agent to the public Blockscout MCP endpoint to fetch the latest block number. |
| **4** | Implement Gas Spike Detection | Fetch recent gas prices, calculate an average/standard deviation, and flag anomalies. |
| **5** | Build Gas Spike Explanation | If a spike is detected, query Blockscout for the top gas-using contracts and report them as the cause. |
| **6** | **Explain Price Movers (Core Feature)** | Combine Blockscout data (large transactions) with Tavily news searches to create a synthesized explanation for token price movements. |
| **7** | (Stretch) Add Opportunity ID | Scan for PoolCreated events on Uniswap via Blockscout to identify new token pairs. |
| **8** | Prepare & Submit Project | Record a video demo, write a great README.md, and clean up the code for submission. |

### **⏩ In Progress**

| ID | Task | Details |
| :---- | :---- | :---- |
|  | *(Empty)* |  |

### **✅ Done**

| ID | Task | Details |
| :---- | :---- | :---- |
|  | *(Empty)* |  |

## **📝 Notes & Collaboration**

* **Next Step:** The first logical task is **ID \#1: Deploy Basic Agentverse Agent**. This ensures the foundational pieces are working before we add any complexity.  
* **Question:** Should we focus on a specific blockchain first (e.g., Ethereum Mainnet) to simplify the initial logic?  
* **Idea:** For the demo, we could prepare a list of interesting recent events (e.g., a major token dump, a viral NFT mint) to showcase the agent's capabilities effectively.