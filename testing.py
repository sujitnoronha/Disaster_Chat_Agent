import os
import getpass
from prompter_245 import prompt

#os.environ["OPENAI_API_KEY"] = getpass.getpass("#")

from langchain.agents import Tool, AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI 
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.agents import AgentExecutor
from langchain import hub
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.tools.render import render_text_description
import gradio as gr
import asyncio
from gradio_tools.tools.gradio_tool import GradioTool

search = DuckDuckGoSearchAPIWrapper()

search_tool = Tool(name="Current Search",
                   func=search.run,
                   description="Useful when you need to answer questions about nouns, current events or the current state of the world."
                   )

tools = [search_tool]

# Set up memory
memory = ConversationBufferMemory(memory_key="chat_history")

# Initialize LLM
llm = ChatOpenAI(model = "gpt-3.5-turbo", temperature=0, openai_api_key="#" )
print('llm initialized')

# Initialize agent
agent_chain = initialize_agent(tools,
                               llm,
                               agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                               memory=memory,
                               verbose=True)

# Run queries
# Here is an example query:
#agent_chain.run(input="What it do, nephew!")
agent_prompt = hub.pull("hwchase17/react-chat")

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

class AgentExecutorInterface(gr.Interface):
    def __init__(self, agent_executor, title=None, description=None):
        inputs = gr.Textbox(lines=5, label="Enter your text here")
        outputs = gr.Textbox(label="Response will appear here")
        super().__init__(self.interact_with_agent, inputs=inputs, outputs=outputs, title=title, description=description)
        self.agent_executor = agent_executor

    def interact_with_agent(self, input_text):
        payload = {"input": input_text}
        return self.agent_executor.invoke(payload)["output"]

    # Example usage

agent_interface = AgentExecutorInterface(agent_executor, title="Agent Executor", description="Interact with the agent executor using this interface")
    

agent_interface.launch()