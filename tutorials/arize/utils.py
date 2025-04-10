from dataclasses import dataclass

import union
from mashumaro.mixins.json import DataClassJSONMixin

VectorDB = union.Artifact(name="milvus-db")


@dataclass
class EmbeddingConfig(DataClassJSONMixin):
    model_name: str = "Snowflake/snowflake-arctic-embed-l-v2.0"
    dimensions: int = 1024


@dataclass
class VectorStoreConfig(DataClassJSONMixin):
    nlist: int = 64
    nprobe: int = 16
    index_type: str = "FLAT"
    collection_name: str = "milvus_arize"
