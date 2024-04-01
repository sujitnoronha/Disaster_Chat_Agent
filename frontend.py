import requests
import gradio as gr

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

# Creating the Gradio interface
iface = gr.Interface(fn=query_fastapi,
                     inputs=gr.inputs.Textbox(lines=2, placeholder="Enter your query here..."),
                     outputs="text",
                     title="Agent Query Interface",
                     description="This interface sends queries to a FastAPI backend and displays the responses.")

# Running the Gradio app
iface.launch()