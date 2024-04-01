# Disaster_Chat_Agent

This information system effectively addresses user queries during disasters. First, a Large Language Model (LLM) analyzes the user’s question to determine if a specialized tool can provide the most accurate response. 
* For real-time disaster information, the system leverages FEMA’s live database.
* RAG QA, a dedicated tool, tackles questions about general disaster preparedness and response procedures. 
* Location-based disaster locators are employed for queries seeking geographically specific resource information. 


The chosen tool retrieves relevant data to craft a response, which the LLM then summarizes for user-friendliness.
Beyond basic summarization, the system utilizes
leverages  reAct prompting.This approach goes beyond simply feeding the summarized information to the LLM. It also incorporates a "scratchpad" that stores the entire conversation history. This allows the LLM to consider past user interactions and context. 
For instance, if a user asks "Where can I find a resource centre?" followed by "Are there any centre near me?", the
LLM can understand the user’s location based on
the conversation history and leverage a location-
based tool to provide a more specific answer in the
second query. This enriched context, facilitated by
reAct prompting, empowers the LLM to generate
informative and relevant responses that directly address the user’s specific needs within the disaster
scenario. The system also handles non-disaster-related inquiries by providing predefined responses that may
direct users to appropriate resources.


## Usage
### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/sujitnoronha/Disaster_Chat_Agent.git
    ```

Built using Lanchain, gpt 3.5 turbo, 



Make sure to replace `YOUR_OPENAI_API_KEY` with your actual OpenAI API key in the `backend_api.py` file before running the backend API. Adjust any other details or instructions as needed for your project.
