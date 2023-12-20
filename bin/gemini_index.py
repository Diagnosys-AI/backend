from dotenv import load_dotenv
import numpy as np
from trulens_eval import Feedback, TruLlama
from trulens_eval.feedback import Groundedness
from trulens_eval.feedback.provider.openai import OpenAI as OpenAITru
from llama_index import SimpleDirectoryReader, load_index_from_storage, VectorStoreIndex, ServiceContext, StorageContext
from llama_index.embeddings import GeminiEmbedding
from llama_index.llms import Gemini
import logging
import os

load_dotenv()

STORAGE_DIR = "./storage"  # directory to cache the generated index
DATA_DIR = "./data"  # directory containing the documents to index

embed_model = GeminiEmbedding(model_name="models/embedding-001")
service_context = ServiceContext.from_defaults(llm=Gemini(), embed_model=embed_model)


def get_index():
    logger = logging.getLogger("uvicorn")
    # check if storage already exists
    if not os.path.exists(STORAGE_DIR):
        logger.info("Creating new index")
        # load the documents and create the index
        documents = SimpleDirectoryReader(DATA_DIR).load_data()
        index = VectorStoreIndex.from_documents(documents, service_context=service_context)
        # store it for later
        index.storage_context.persist(STORAGE_DIR)
        logger.info(f"Finished creating new index. Stored in {STORAGE_DIR}")
    else:
        # load the existing index
        logger.info(f"Loading index from {STORAGE_DIR}...")
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        index = load_index_from_storage(storage_context, service_context=service_context)
        logger.info(f"Finished loading index from {STORAGE_DIR}")
    return index


index = get_index()
query_engine = index.as_query_engine()


resp = query_engine.query("When was the University of Washington founded?")
print(resp)



# Initialize provider class
openai_tru = OpenAITru()
grounded = Groundedness(groundedness_provider=OpenAITru())
# Define a groundedness feedback function
f_groundedness = Feedback(grounded.groundedness_measure_with_cot_reasons).on(
    TruLlama.select_source_nodes().node.text.collect()
    ).on_output(
    ).aggregate(grounded.grounded_statements_aggregator)
# Question/answer relevance between overall question and answer.
f_qa_relevance = Feedback(openai_tru.relevance).on_input_output()

# Question/statement relevance between question and each context chunk.
f_qs_relevance = Feedback(openai_tru.qs_relevance).on_input().on(
    TruLlama.select_source_nodes().node.text
    ).aggregate(np.mean)

tru_query_engine_recorder = TruLlama(query_engine,
    app_id='Diagnosys',
    feedbacks=[f_groundedness, f_qa_relevance, f_qs_relevance])

with tru_query_engine_recorder as recording:
    resp = query_engine.query("When was the University of Washington founded?")
    print(resp)