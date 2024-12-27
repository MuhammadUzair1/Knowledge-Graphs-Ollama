from pydantic import BaseModel
from typing import List, Dict, Optional


class Chunk(BaseModel):
    chunk_id: int
    content: str
    embedding: Optional[List[float]] = None
    chunk_size: int=1000
    chunk_overlap: int=100
    embeddings_model: Optional[str] = None


class ProcessedDocument(BaseModel):
    filename: str = ""
    text: str = ""
    document_version: int = 1
    metadata: Optional[dict] = None
    source: str = "local_directory"
    chunks: Optional[List[Chunk]] = None


class Ontology(BaseModel):
    """ Ontology of the graph DB to be built or queried """
    allowed_labels: Optional[List[str]]=None
    labels_descriptions: Optional[Dict[str, str]]=None
    allowed_relations: Optional[List[str]]=None