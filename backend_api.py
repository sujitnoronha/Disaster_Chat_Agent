from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
import getpass
from prompter_245 import prompt
from langchain.agents import Tool, AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI 
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain.agents import AgentExecutor
from langchain import hub
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.tools.render import render_text_description
import gradio as gr
from utils import fema_tool, rag_tool, DisasterRecoverySearchTool, rate_tool


open_ai_key = "" #put key here

# Assuming the AgentExecutor and all necessary setup from the notebook is already initialized above this point

app = FastAPI()

search = DuckDuckGoSearchAPIWrapper()

search_tool = Tool(name="Current Search",
                   func=search.run,
                   description="Useful when you need to answer questions about current events or the current state of the world.")


tools = [search_tool, fema_tool, rag_tool, DisasterRecoverySearchTool(), rate_tool]

# Set up memory
memory = ConversationBufferMemory(memory_key="chat_history")

# Initialize LLM
llm = ChatOpenAI(model = "gpt-3.5-turbo", temperature=0, openai_api_key=open_ai_key )
print('llm initialized')

# Initialize agent
agent_chain = initialize_agent(tools,
                               llm,
                               agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                               memory=memory,
                               verbose=True)

agent_prompt = prompt

# Setting up an agent using LCEL
prompt = agent_prompt.partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools]),
)
llm_with_stop = llm.bind(stop=["\nObservation"])

# Set up the agent using LCEL
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | llm_with_stop
    | ReActSingleInputOutputParser()
)

memory = ConversationBufferMemory(memory_key="chat_history")

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory)


class AgentQuery(BaseModel):
    input: str

@app.post("/query/")
async def query_agent(query: AgentQuery):
    try:
        response = agent_executor.invoke({"input": query.input})
        print(response['output'])
        return {"output": response["output"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)