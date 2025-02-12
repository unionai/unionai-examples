import flytekit as fl

nvidia_key = "nvidia-build-api-key"
eleven_key = "elevenlabs-api-key"

image = fl.ImageSpec(
    name="pdf-to-podcast",
    packages=[
        "pydantic==2.10.5",
        "langchain_nvidia_ai_endpoints==0.3.7",
        "langchain-core==0.3.29",
        "ujson==5.10.0",
        "docling==2.2.0",
    ],
).with_commands(
    [
        'python -c "from deepsearch_glm.utils.load_pretrained_models import load_pretrained_nlp_models; load_pretrained_nlp_models(verbose=True)"',
    ]
)
