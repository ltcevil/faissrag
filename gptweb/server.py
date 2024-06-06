from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import RedirectResponse
from langserve import add_routes
from rag_faiss import chain as rag_gpt_crawler
from rag_faiss import embeddings

app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")



add_routes(app, rag_gpt_crawler, path="/rag")


@app.post("/generate-embeddings")
async def generate_embeddings_endpoint(background_tasks: BackgroundTasks):
    background_tasks.add_task(embeddings.genembeddings)
    return {"message": "Embedding generation started in the background"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
