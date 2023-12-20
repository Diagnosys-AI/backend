from typing import List

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from app.utils.json_model import json_to_model
from app.utils.index import get_index
from llama_index import VectorStoreIndex
from llama_index.llms import ChatMessage
from llama_index.llms.types import MessageRole

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
    You are an AI medical advisor who integrates a comprehensive array of medical resources for verifying information and assumptions. These include PubMed, CDC, WHO, ClinicalTrials.gov, UpToDate, Mayo Clinic, Cleveland Clinic, AMA, NIH, BMJ, The Lancet, JAMA, Cochrane Library, Medscape, WebMD, NCBI, ScienceDirect, EMBASE, PLOS Medicine, Nature Medicine, Cell, MDPI, Radiopaedia, PsychINFO, BioMed Central, ACP, and NEJM. You are committed to continually expanding your use of resources, aiming to utilize the full breadth of these tools and incorporate new and better ones as they become available. This ensures that you provide the most up-to-date, evidence-based medical information and advice, drawing from a wide range of reputable and peer-reviewed sources. You have been provided a patient history, that includes the following information:

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
    
    Provide a list of direct, straightforward list of potential diagnoses, investigations and treatments for this presenting condition. Use the History of Presenting Complaint, Past Medical History, Medication History, Social History and Family History to inform your suggestions. You can use the following resources to help you:
    
    - The Oxford Handbook of Clinical Medicine is a pocket textbook aimed at medical students and junior doctors, and covers all aspects of clinical medicine.
    - Kumar & Clark's Clinical Medicine - Textbook aimed at medical students in the preclinical years of study.
    - DermNet NZ - DermNet NZ is a website dedicated to dermatology. It has a large collection of photos and information on skin conditions.
    - NICE CKS - NICE guidelines are evidence-based recommendations for health and care in England. They set out the care and services suitable for most people with a specific condition or need, and people in particular circumstances or settings.
    """
    return "tell me a joke"

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

    query_engine = index.as_query_engine(
        similarity_top_k=1,
        streaming=True
    )
    response = query_engine.query(
        lastMessage.content
    )

    # query chat engine
    # chat_engine = index.as_chat_engine()

    # chat_engine = Gemini()
    # response = chat_engine.stream_chat(messages)

    # stream response
    async def event_generator():
        for token in response.response_gen:
            # If client closes connection, stop sending events
            if await request.is_disconnected():
                break
            yield token

    return StreamingResponse(event_generator(), media_type="text/plain")
