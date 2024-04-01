import os
from typing import Optional, Tuple
import requests
import gradio as gr
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from threading import Lock
import os
import getpass
from prompter_245 import prompt

#os.environ["OPENAI_API_KEY"] = getpass.getpass("")

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


def load_chain():
    """Logic for loading the chain you want to use should go here."""
    llm = OpenAI(temperature=0)
    chain = ConversationChain(llm=llm)
    return chain


def set_openai_api_key(api_key: str):
    """Set the api key and return chain.

    If no api_key, then None is returned.
    """
    if api_key:
        os.environ["OPENAI_API_KEY"] = "" #put api key here
        chain = load_chain()
        return chain

def query_fastapi(input_text):
    # URL of the FastAPI endpoint
    url = "http://localhost:8000/query/"
    # Sending a POST request to the FastAPI server with the input
    response = requests.post(url, json={"input": input_text})
    # Extracting the output from the response
    if response.status_code == 200:
        output = response.json().get("output", "No output received")
    else:
        output = f"Error: {response.text}"
    return output

class ChatWrapper:

    def __init__(self):
        self.lock = Lock()
    def __call__(
        self, api_key: str, inp: str, history: Optional[Tuple[str, str]], chain: Optional[ConversationChain]
    ):
        """Execute the chat functionality."""
        self.lock.acquire()
        try:
            history = history or []
            # If chain is None, that is because no API key was provided.
            if chain is None:
                history.append((inp, "Please paste your OpenAI key to use"))
                return history, history
            #Set OpenAI key
            
            import openai
            openai.api_key = api_key
            # Run chain and append input.
            
            output = query_fastapi(inp)
            history.append((inp, output))
        except Exception as e:
            raise e
        finally:
            self.lock.release()
        return history, history

chat = ChatWrapper()

block = gr.Blocks(css=".gradio-container {background-color: lightgray}")

with block:
    with gr.Row():
        gr.Markdown("<h1><center>DICE (Disaster Intervention Chatbot Engine)</center></h1>")

        openai_api_key_textbox = gr.Textbox(
            placeholder="Paste your OpenAI API key (sk-...)",
            show_label=False,
            lines=1,
            type="password",
        )

    chatbot = gr.Chatbot()

    with gr.Row():
        message = gr.Textbox(
            label="What's your question?",
            placeholder="Ask any question related to natural disasters",
            lines=1,
        )
        submit = gr.Button(value="Send", variant="secondary").style(full_width=False)

    gr.Examples(
        examples=[
            "There's a wildfire what should i do?",
            "Can you help me with disaster recovery centers near me?",
            "What are some things I should keep in mind in case of a wildfire?",
        ],
        inputs=message,
    )

    gr.HTML("DICE AGI.")

    gr.HTML(
        "<center>Powered by AI TOOLS</center>"
    )

    state = gr.State()
    agent_state = gr.State()

    submit.click(chat, inputs=[openai_api_key_textbox, message, state, agent_state], outputs=[chatbot, state])
    message.submit(chat, inputs=[openai_api_key_textbox, message, state, agent_state], outputs=[chatbot, state])

    openai_api_key_textbox.change(
        set_openai_api_key,
        inputs=[openai_api_key_textbox],
        outputs=[agent_state],
    )

block.launch(debug=True)