from typing import List

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from app.utils.json_model import json_to_model
from app.utils.index import get_index
from app.utils.tru_query_engine_recorder import get_tru_query_engine_recorder
from llama_index import VectorStoreIndex
from llama_index.llms import ChatMessage
from llama_index.llms.types import MessageRole
import io

from pydantic import BaseModel

chat_router = r = APIRouter()


class _Message(BaseModel):
    role: MessageRole
    content: str
    data: dict | None = None


class _ChatData(BaseModel):
    messages: List[_Message]

def format_history_prompt(formData):
    prompt = f"""
    You are a medical advisor who integrates a comprehensive array of medical resources for verifying information and assumptions. You have been provided a patient history, that includes the following information:

    1. **Introduction:**: {formData['introduction']}
    2. **Presenting Complaint (PC):** {formData['presentingComplaint']}
    3. **History of Presenting Complaint (HxPC):**
       - **SOCRATES:** {formData['socrates']}
       - **Specific Systems Review:** {formData['specificSystemsReview']}
       - **General Systems Review:** {formData['generalSystemsReview']}
       - **ICE:** {formData['ice']}
    4. **Past Medical History (PMHx):** {formData['pastMedicalHistory']}
    5. **Medication History (MHx):** {formData['medicationHistory']}
    6. **Social History (SHx):**: {formData['socialHistory']}
    7. **Family History (FHx):** {formData['familyHistory']}
    
    Provide a list of Management and Further treatment options for this patient.
    """
    return prompt

@r.post("")
async def chat(
    request: Request,
    # Note: To support clients sending a JSON object using content-type "text/plain",
    # we need to use Depends(json_to_model(_ChatData)) here
    data: _ChatData = Depends(json_to_model(_ChatData)),
    index: VectorStoreIndex = Depends(get_index),
):
    # check preconditions and get last message
    if len(data.messages) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No messages provided",
        )
    # if data.form is true in any of the messages, replace content with "tell me a joke"
    for m in data.messages:
        if m.data is not None and m.data["form"] is True:
            m.content = format_history_prompt(m.data["formData"])


    lastMessage = data.messages[-1]
    if lastMessage.role != MessageRole.USER:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Last message must be from user",
        )
    # convert messages coming from the request to type ChatMessage
    messages = [
        ChatMessage(
            role=m.role,
            content=m.content,
        )
        for m in data.messages
    ]

    # query chat engine
    # chat_engine = index.as_chat_engine()
    # chat_engine = Gemini()
    # response = chat_engine.stream_chat(messages)

    query_engine = index.as_query_engine(
    )
    tru_query_engine_recorder = get_tru_query_engine_recorder(query_engine)

    with tru_query_engine_recorder as recording:
        response = query_engine.query(
            lastMessage.content
        )

    # Convert Response string into a Stream
    # Mock because TruLens doesnt work well with streaming a Gemini Index
    stream = io.StringIO(str(response) + "\n")

    # async def event_generator():
    #     for token in response.response_gen:
    #         # If client closes connection, stop sending events
    #         if await request.is_disconnected():
    #             break
    #         yield token

    return StreamingResponse(stream, media_type="text/plain")
