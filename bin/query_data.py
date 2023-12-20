from llama_index import VectorStoreIndex, SimpleDirectoryReader
from dotenv import load_dotenv

# Make sure your API key is available to your code by setting it as an environment variable.
load_dotenv()
# LlamaIndex uses OpenAI’s gpt-3.5-turbo model by default.
documents = SimpleDirectoryReader("../data").load_data()
# This builds an index over the documents in the data folder (which in this case just consists of the essay text, but could contain many documents).
index = VectorStoreIndex.from_documents(documents)
# This creates an engine for Q&A over your index and asks a simple question.
query_engine = index.as_query_engine()
response = query_engine.query("What are Dimensional Standards for Letters?")
print(response)