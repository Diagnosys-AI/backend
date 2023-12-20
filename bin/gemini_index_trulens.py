from app.utils.index import get_index
from dotenv import load_dotenv

import numpy as np

from trulens_eval.feedback.provider.openai import OpenAI as OpenAI_Trulens
from trulens_eval import Feedback, TruLlama
from trulens_eval.feedback import Groundedness

load_dotenv()

index = get_index(data_dir="./data_tmp")
query_engine = index.as_query_engine()

# Initialize provider class
openai_tru = OpenAI_Trulens()
grounded = Groundedness(groundedness_provider=OpenAI_Trulens())

# Define a Groundedness feedback function
# Is the response supported by the context?
f_groundedness = (Feedback(grounded.groundedness_measure_with_cot_reasons)
    .on(TruLlama.select_source_nodes().node.text.collect())
    .on_output()
    .aggregate(grounded.grounded_statements_aggregator))
# Question/answer relevance between overall question and answer.
# Is the answer relevant to the query?
f_qa_relevance = Feedback(openai_tru.relevance).on_input_output()

# Question/statement relevance between question and each context chunk.
# Is the retrieved context relevant to the query?
f_qs_relevance = Feedback(openai_tru.qs_relevance).on_input().on(
    TruLlama.select_source_nodes().node.text
    ).aggregate(np.mean)

tru_query_engine_recorder = TruLlama(query_engine,
    app_id='Diagnosys',
    feedbacks=[f_groundedness, f_qa_relevance, f_qs_relevance])

with tru_query_engine_recorder as recording:
    resp = query_engine.query("Tell me about UW")
    print(resp)
