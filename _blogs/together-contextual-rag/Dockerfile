FROM hamersaw/wolf:v0

USER root
WORKDIR /root
ENV PYTHONPATH /root

RUN apt-get update && apt-get install -y git

RUN pip install together beautifulsoup4 bm25s pydantic>2 chromadb

USER flytekit