import pandas as pd
from geopy.distance import great_circle
import requests
from langchain.tools import Tool  # Assuming Tool is correctly imported from LangChain or defined elsewhere
from sentence_transformers import SentenceTransformer
import faiss
import torch

from typing import Optional, Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun



class TextLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        with open(self.file_path, 'r') as file:
            documents = file.readlines()
        return [{"page_content": doc.strip()} for doc in documents]


loader = TextLoader("Disaster_RAG.txt")
documents = loader.load()
text_documents = [doc["page_content"] for doc in documents]


model = SentenceTransformer("sentence-transformers/all-MiniLM-l6-v2")
embeddings = model.encode(text_documents, convert_to_tensor=True)

# Create FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
faiss.normalize_L2(embeddings.numpy())
index.add(embeddings.numpy())


def generate_augmented_answer(query):
    # Load text documents

    # Embed the documents

    # Embed the query
    query_embedding = model.encode([query], convert_to_tensor=True)
    faiss.normalize_L2(query_embedding.numpy())

    # Perform similarity search
    _, doc_indices = index.search(query_embedding.numpy(), k=5)
    similar_docs = [text_documents[i] for i in doc_indices[0]]

    # Augment the query with information from similar documents
    augmented_query = "Considering relevant information from retrieved documents: " + "\n  ".join(similar_docs) + ", what is the answer to the question: " + query

    # Query the text generation model
    # from transformers import pipeline
    # generator = pipeline("text-generation", model="google/flan-t5-base", tokenizer="google/flan-t5-base", framework="pt")
    # augmented_answers = generator(augmented_query, max_length=256, temperature=1)

    return augmented_query




# Load the data from the CSV
df = pd.read_csv("disaster_recovery_centers.csv")

def find_nearest_service(user_lat, user_long):
    """
    Finds the nearest disaster recovery center to the user's location.

    Parameters:
    user_lat (float): The latitude of the user's location.
    user_long (float): The longitude of the user's location.

    Returns:
    dict: Information about the nearest center including a Google Maps link.
    """
    # Calculate distances from user location to all centers
    user_location = (user_lat, user_long)
    df['Distance'] = df.apply(lambda row: great_circle(user_location, (row['Latitude'], row['Longitude'])).miles, axis=1)

    # Find the nearest center
    nearest_center = df.loc[df['Distance'].idxmin()]

    # Construct Google Maps link for the nearest center
    maps_link = f"https://www.google.com/maps/search/?api=1&query={nearest_center['Latitude']},{nearest_center['Longitude']}"

    # Return relevant information about the nearest center
    result = {
        "Location Name": nearest_center['Location Name'],
        "Description": nearest_center['Description'],
        "Services Provided": nearest_center['Services Provided'],
        "Google Maps Link": maps_link
    }



    return result




# Original function adapted to accept zip code as an argument
def fetch_and_process_disasters(zip_code):
    def fetch_disasters():
        url = "https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data['DisasterDeclarationsSummaries']
        else:
            print(f"Failed to fetch data: {response.status_code}")
            return []

    def filter_disasters_by_place_code(disasters, place_code):
        return [disaster for disaster in disasters if str(disaster.get('placeCode', '')) == str(place_code)]

    all_disasters = fetch_disasters()
    relevant_disasters = filter_disasters_by_place_code(all_disasters, zip_code)
    
    if not relevant_disasters:
        return "No disasters found for this place code."

    output = []
    for disaster in relevant_disasters:
        disaster_info = f"Title: {disaster.get('declarationTitle', 'N/A')}, " \
                        f"State: {disaster.get('state', 'N/A')}, " \
                        f"Begin Date: {disaster.get('incidentBeginDate', 'N/A')}, " \
                        f"End Date: {disaster.get('incidentEndDate', 'N/A')}"
        output.append(disaster_info)
    return "\n".join(output)



# Wrapper function for compatibility with LangChain
def fetch_and_process_disasters_wrapper(input_text):
    # Directly use the input text as the zip code for simplicity
    return fetch_and_process_disasters(input_text)

class DisasterRecoverySearchInput(BaseModel):
    lat_long: str = Field(description="Latitude and Longitude details (USE latitude 32.0522 longitude: 119.2437)")

# Custom tool for searching the nearest disaster recovery center
class DisasterRecoverySearchTool(BaseTool):
    name = "disaster_recovery_search"
    description = "Finds the nearest disaster recovery center based on the user's location "
    args_schema: Type[BaseModel] = DisasterRecoverySearchInput

    def _run(
        self, lat_long: str,  run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> dict:
        """Use the tool to find the nearest disaster recovery center."""
        
        return find_nearest_service(32.0522, 119.2437)

    async def _arun(
        self, lat_long: str,  run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> dict:
        """Use the tool asynchronously to find the nearest disaster recovery center."""
        # Asynchronous implementation would go here if applicable.
        # For now, we'll raise a NotImplementedError as async is not supported for this tool.
        raise NotImplementedError("DisasterRecoverySearchTool does not support async")



rag_tool = Tool(
    name='RAG Search',
    func=generate_augmented_answer,  # Use the wrapper function here
    description="Use this information to answer help related to questions for Natural Disasters such as, earthquake, wildfires, pandemics, etc. provide details such as the phone numbers or location details. (NOTE: Should not be used to answer questions related to current events use the current search tool in that case)",
)


fema_tool = Tool(
    name='FEMA Search',
    func=fetch_and_process_disasters,  # Use the wrapper function here
    description="Useful when asking zipcode based current disaster details"
)


def ask_for_rating(input_text):

        return "On a scale of 1 to 10, how adequately do you feel your needs were met during the disaster response?"


rate_tool = Tool(
    name='Rate Experience',
    func=ask_for_rating,  # Use the function to handle rating response
    description="Useful when the user says, thats all or THANKS it for helping, ask for rating response"
)


