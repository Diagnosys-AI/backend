import logging
import os
from llama_index import SimpleDirectoryReader, load_index_from_storage, VectorStoreIndex, ServiceContext, StorageContext
from llama_index.embeddings import GeminiEmbedding
from llama_index.llms import Gemini

def get_index(
        storage_dir: str = "./storage",
        data_dir: str = "./md",
):
    logger = logging.getLogger("uvicorn")
    embed_model = GeminiEmbedding(model_name="models/embedding-001")
    service_context = ServiceContext.from_defaults(llm=Gemini(), embed_model=embed_model)
    # check if storage already exists
    if not os.path.exists(storage_dir):
        logger.info("Creating new index")
        # load the documents and create the index
        documents = SimpleDirectoryReader(data_dir).load_data()
        # Using the embedding model to Gemini
        index = VectorStoreIndex.from_documents(documents, service_context=service_context)
        # store it for later
        index.storage_context.persist(storage_dir)
        logger.info(f"Finished creating new index. Stored in {storage_dir}")
    else:
        # load the existing index
        logger.info(f"Loading index from {storage_dir}...")
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        index = load_index_from_storage(storage_context,service_context=service_context)
        logger.info(f"Finished loading index from {storage_dir}")
    return index



if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    index = get_index()
    query_engine = index.as_query_engine()
    resp = query_engine.query("What can you tell me about UW?")
    print(resp)

