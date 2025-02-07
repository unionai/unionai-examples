import os

import gradio as gr
from openai import OpenAI
from openinference.instrumentation.openai import OpenAIInstrumentor
from phoenix.otel import register

os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"api_key={os.getenv('PHOENIX_API_KEY')}"
os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={os.getenv('PHOENIX_API_KEY')}"
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com"

tracer_provider = register()

OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)


def stream_response(query, history):
    history.append({"role": "user", "content": query})

    client = OpenAI(
        base_url=f"{os.getenv('VLLM_DEEPSEEK_ENDPOINT')}/v1",
        api_key="random",
    )

    response = client.chat.completions.create(
        model="NousResearch/Meta-Llama-3-8B-Instruct", messages=history, stream=True
    )

    for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            yield content


demo = gr.ChatInterface(
    stream_response, type="messages", title="DeepSeek R1 Distill Qwen 1.5B Chatbot"
)

demo.launch(server_port=8080)
