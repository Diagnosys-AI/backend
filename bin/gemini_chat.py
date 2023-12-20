from llama_index.llms import ChatMessage, Gemini
from dotenv import load_dotenv

# Make sure your API key is available to your code by setting it as an environment variable.
load_dotenv()

# Calling Complete

# resp = Gemini().complete("Write a poem about a magic backpack")
# print(resp)

# Call Chat
# messages = [
#     ChatMessage(role="user", content="Hello friend!"),
#     ChatMessage(role="assistant", content="Yarr what is shakin' matey?"),
#     ChatMessage(
#         role="user", content="Help me decide what to have for dinner."
#     ),
# ]
# resp = Gemini().chat(messages)
# print(resp)


# Stream Chat for UI
llm = Gemini()
messages = [
    ChatMessage(role="user", content="Hello friend!"),
    ChatMessage(role="assistant", content="Yarr what is shakin' matey?"),
    ChatMessage(
        role="user", content="Help me decide what to have for dinner."
    ),
]
resp = llm.stream_chat(messages)
for r in resp:
    print(r.delta, end="")