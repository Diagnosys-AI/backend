from dotenv import load_dotenv
from trulens_eval import Feedback, Tru, TruLlama
from trulens_eval.feedback import Groundedness
from llama_index import VectorStoreIndex
from llama_index.readers.web import SimpleWebPageReader
from trulens_eval.feedback.provider.openai import OpenAI
import numpy as np

load_dotenv()

tru = Tru()

documents = SimpleWebPageReader(
    html_to_text=True
).load_data(["http://paulgraham.com/worked.html"])
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()
# Initialize provider class
openai = OpenAI()

grounded = Groundedness(groundedness_provider=OpenAI())

# Define a groundedness feedback function
f_groundedness = Feedback(grounded.groundedness_measure_with_cot_reasons).on(
    TruLlama.select_source_nodes().node.text.collect()
    ).on_output(
    ).aggregate(grounded.grounded_statements_aggregator)

# Question/answer relevance between overall question and answer.
f_qa_relevance = Feedback(openai.relevance).on_input_output()

# Question/statement relevance between question and each context chunk.
f_qs_relevance = Feedback(openai.qs_relevance).on_input().on(
    TruLlama.select_source_nodes().node.text
    ).aggregate(np.mean)

tru_query_engine_recorder = TruLlama(query_engine,
    app_id='OpenAI TruLens',
    feedbacks=[f_groundedness, f_qa_relevance, f_qs_relevance])

# or as context manager
with tru_query_engine_recorder as recording:
    resp = query_engine.query("What did the author do growing up?")
    print(resp)