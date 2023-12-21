from app.utils.index import get_index
from dotenv import load_dotenv

from llama_index.llms import ChatMessage

import numpy as np

from trulens_eval.feedback.provider.openai import OpenAI as OpenAI_Trulens
from trulens_eval import Feedback, TruLlama
from trulens_eval.feedback import Groundedness
load_dotenv()

index = get_index(data_dir="../md")
chat_engine = index.as_chat_engine()

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

tru_query_engine_recorder = TruLlama(chat_engine,
    app_id='Diagnosys',
    feedbacks=[f_groundedness, f_qa_relevance, f_qs_relevance])

with (tru_query_engine_recorder as recording):
    data = [
        {
            "role": "user",
            "content": "Tell me about UW"
        },
        {
            "role": "system",
            "content": "The University of Washington (UW) is a public research university founded in 1861. It has over 45,000 students across three campuses in Seattle, Tacoma, and Bothell. As the flagship institution of the six public universities in Washington state, UW encompasses over 500 buildings and 20 million square feet of space, including one of the largest library systems in the world."
        },
        {
            "role": "user",
            "content": "What year and in what city was The University of Washington founded? Also, how many students does it have?"
        }
    ]
    lastMessage = data.pop()
    messages = [
        ChatMessage(
            role=m["role"],
            content=m["content"],
        )
        for m in data
    ]
    resp = chat_engine.chat(message=lastMessage["content"], chat_history=messages)
    # Print Streamed Response
    for token in resp.response_gen:
        print(token)

