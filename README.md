# Diagnosys Fast API

This is a [LlamaIndex](https://www.llamaindex.ai/) project using [FastAPI](https://fastapi.tiangolo.com/).

## Getting Started

First, setup the environment:

```
poetry install
poetry shell
```

Second, run the development server:

```
python main.py
```

Then call the API endpoint `/api/chat` to see the result:

```
curl --location 'localhost:8000/api/chat' \
--header 'Content-Type: application/json' \
--data '{ "messages": [{ "role": "user", "content": "Hi" }] }'
```

## Docs

Open [http://localhost:8000/docs](http://localhost:8000/docs) with your browser to see the Swagger UI of the API. The API allows CORS for all origins to simplify development. You can change this behavior by setting the `ENVIRONMENT` environment variable to `prod`:

```
ENVIRONMENT=prod uvicorn main:app
```

## TruLens

We recommended using TruLens by TruEra for optimization and explainability. Enhance your LLM apps with TruLens: Use sophisticated evaluation and performance tracking tools for superior output quality. Good for RAGs and no-hallucination scenarios.

### TruLens-Eval

Enhance your LLM-based applications with TruLens by assessing inputs, outputs, and internal mechanics. Benefit from feedback on groundedness, relevance, and toxicity.

### TruLens-Explain

Monitor your application's performance with TruLens' instrumentation, useful for various LLM applications. Track usage metrics and metadata for deeper insights.

## Links

- [Hackathon Info](https://lablab.ai/event/gemini-ai-hackathon)
- [LlamaIndex: a Data Framework for LLM Applications](https://lablab.ai/tech/llamaindex)
  - [GitHub](https://github.com/run-llama/llama_index)
- [TruLens Notebook](https://github.com/truera/trulens/blob/main/trulens_eval%2Fexamples%2Fexpositional%2Fmodels%2Fgoogle_vertex_quickstart.ipynb)
- [Gemini Notebook](https://colab.research.google.com/drive/11b6-GvwIXB5r_qsoLAYEa6ejIPUoC2hw)
- [Multi-Modal LLM using Google's Gemini model for image understanding and build](https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/examples/multi_modal/gemini.ipynb)
- [Gemini Example](https://colab.research.google.com/drive/1W5UkUpjNK9wi7RzuEv83vC0ejBgJ5Ty3?usp=sharing)

## Criteria

- **Usage of Technology:** Effectiveness of integrating Gemini AI and TruLens into the project.
- **Completion of Poc:** Degree to which the proof of concept is fully developed and functional.
- **Technical Scalability:** Ability to handle increased load without compromising performance.
- **Usability:** Intuitiveness and aesthetic appeal of the application's interface.
- **Business Plan:** Commercial viability and market potential of the project.
- **Presentation:** Effectiveness in communicating the project's ideas clearly.
- **Team Structure:** Efficiency and balance in team organization and roles.

## Educational Materials

Resources that will familiarize you with Gemini AI

- [Gemini AI Documentation](https://ai.google.dev/docs)
- [Test and Get Started with Gemini AI on Google Cloud Vertex AI](https://console.cloud.google.com/vertex-ai/generative/multimodal/prompt-examples/Extract%20text%20from%20images?project=wdll-252515)
- [Gemini Multimodal Prompting Guide](https://developers.googleblog.com/2023/12/how-its-made-gemini-multimodal-prompting.html)
- [Video Guide from Sam Witteveen: Getting Started with Gemini Pro on Google AI Studio](https://www.youtube.com/watch?v=HN96QDFBD0g)
- [TruLens Evaluating Multi-Modal RAG Notebook](https://github.com/truera/trulens/blob/main/trulens_eval%2Fexamples%2Fexpositional%2Fframeworks%2Fllama_index%2Fllama_index_multimodal.ipynb)
- [RAG App in Production Blog](https://blog.llamaindex.ai/shipping-your-retrieval-augmented-generation-app-to-production-with-create-llama-7bbe43b6287d)
- [Advanced Multi-Modal Retrieval using GPT4V and Multi-Modal Index/Retriever](https://github.com/run-llama/llama_index/blob/main/docs/examples/multi_modal/gpt4v_multi_modal_retrieval.ipynb)

## Learn More

To learn more about LlamaIndex, take a look at the following resources:

- [LlamaIndex Documentation](https://docs.llamaindex.ai) - learn about LlamaIndex.
