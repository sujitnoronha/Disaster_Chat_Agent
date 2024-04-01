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
informative and relevant responses that directly ad-
dress the user’s specific needs within the disaster
scenario.
The system also handles non-disaster-related in-
quiries by providing predefined responses that may
direct users to appropriate resources. By contin-
uously storing user interaction data, the system
undergoes a learning process. This ongoing analy-
sis improves the LLM’s ability to understand user
intent, select the most appropriate tools, and ulti-
mately deliver effective support during disasters.
This cyclical process ensures the system remains
adaptable and responsive, providing users with the
most accurate and up-to-date information during
critical situations.
