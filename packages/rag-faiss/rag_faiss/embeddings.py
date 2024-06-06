import json
import os
from datetime import datetime
from pathlib import Path
from langchain_community.vectorstores.faiss import FAISS, DistanceStrategy
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings

# Load environment variables
api_key = os.getenv('AZURE_API_KEY')
deployment_embed = os.getenv('AZURE_ADA')
api_version = os.getenv('AZURE_API_VERSION')
azure_endpoint = os.getenv('AZURE_API_ENDPOINT')

# Initialize embeddings
embeddings = AzureOpenAIEmbeddings(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=azure_endpoint,
    model=deployment_embed,
)

# Function to generate embeddings
def genembeddings():
    path_to_gptcrawler = "/code/data/output-1-1.json"

    if os.path.exists(path_to_gptcrawler) and os.path.getsize(path_to_gptcrawler) > 0:
        with open(path_to_gptcrawler, 'r') as f:
            data = json.load(f)
    else:
        print(f"File {path_to_gptcrawler} does not exist or is empty.")
        data = []  # Ensure data is an empty list if the file doesn't exist or is empty

    docs = [
        Document(
            page_content=dict_["html"],
            metadata={"title": dict_["title"], "url": dict_["url"]},
        )
        for dict_ in data
    ]

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_splits = text_splitter.split_documents(docs)

    # Generate FAISS index file name with date-time format
    current_datetime = datetime.now().strftime("%Y-%m-%d-%H")
    index_file = f"/code/faiss/{current_datetime}.faiss"

    # Load or create FAISS index
    index_files = sorted(Path("/code/faiss").glob("*.faiss"))
    if index_files:
        latest_index_file = index_files[-1]
        db = FAISS.load_local(latest_index_file, embeddings, allow_dangerous_deserialization=True)
    else:
        db = FAISS.from_documents(documents=all_splits, embedding=embeddings, distance_strategy=DistanceStrategy.COSINE)
        db.save_local(index_file)

    # Create FAISS vectorstore from documents and embeddings
    db = FAISS.from_documents(documents=all_splits, embedding=embeddings, distance_strategy=DistanceStrategy.COSINE)
    db.save_local(index_file)
    print(f"Embeddings generated and saved to {index_file}")
