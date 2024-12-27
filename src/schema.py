from pydantic import BaseModel
from typing import List, Dict, Optional

from langchain_neo4j.graphs.graph_document import Node, Relationship


class Chunk(BaseModel):
    chunk_id: int
    text: str
    embedding: Optional[List[float]] = None
    chunk_size: int=1000
    chunk_overlap: int=100
    embeddings_model: Optional[str] = None


class ProcessedDocument(BaseModel):
    filename: str = ""
    source: str= ""
    document_version: int = 1
    metadata: Optional[dict] = None
    chunks: Optional[List[Chunk]] = None
    nodes: Optional[List[Node]] = None
    relationships: Optional[List[Relationship]] = None


class Ontology(BaseModel):
    """ Ontology of the graph DB to be built or queried """
    allowed_labels: Optional[List[str]]=None
    labels_descriptions: Optional[Dict[str, str]]=None
    allowed_relations: Optional[List[str]]=None