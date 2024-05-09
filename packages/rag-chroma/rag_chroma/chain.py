import json
import os
from pathlib import Path
from datetime import datetime

from flask_restx import Model
from langchain_community.vectorstores.faiss import FAISS, DistanceStrategy
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.chains.retrieval_qa.base import (RetrievalQA,
                                                StuffDocumentsChain,
                                                VectorDBQA)

model = AzureChatOpenAI(
    api_key="989bb2fa71594e5d8b4b8ee0ab749d0e",
    api_version="2024-03-01-preview",
    azure_endpoint="https://jarinachat.openai.azure.com",
    model="jarinagpt-35-turbo",
    temperature=0.6,
    max_tokens=2048,
    streaming=False,
)

embeddings = AzureOpenAIEmbeddings(
    api_key="989bb2fa71594e5d8b4b8ee0ab749d0e",
    api_version="2024-03-01-preview",
    azure_endpoint="https://jarinachat.openai.azure.com",
    model="jarina-ada",

)

# Load output from gpt crawler
path_to_gptcrawler = "/code/data/output-1.json"


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
# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
all_splits = text_splitter.split_documents(docs)

# Generate FAISS index file name with date-time format
current_datetime = datetime.now().strftime("%Y-%m-%d-%H")
index_file = f"/code/faiss/{current_datetime}.faiss"

# Create FAISS vectorstore from documents and embeddings
db = FAISS.from_documents(documents=all_splits, embedding=embeddings, normalize_L2=True, distance_strategy=DistanceStrategy.COSINE)
db.save_local(index_file)

# RAG prompt
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

hun = "answers are expected to be in Hungarian"
qa_chain = RetrievalQA.from_chain_type(
    model,
    retriever=db.as_retriever(),
    chain_type_kwargs={"prompt": prompt + hun}
)

# RAG chain
chain = (
    RunnableParallel({"context": db.as_retriever(), "question": RunnablePassthrough()})
    | prompt
    | model
    | StrOutputParser()
)

# Add typing for input
class Question(BaseModel):
    __root__: str

chain = chain.with_types(input_type=Question)

