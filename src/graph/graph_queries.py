from typing import List, Optional, Tuple
import networkx as nx
from neo4j import Query, Session

from src.graph.graph_model import Node, Relationship, Community, CommunityReport
from src.graph.knowledge_graph import KnowledgeGraph
from src.schema import Chunk
from src.utils.logger import get_logger


logger = get_logger(__name__)


def document_metadata(session: Session, filename: str, version: Optional[int]) -> dict:
    """ Returns a dictionary with metadata from a `Document` node in the Graph"""
    pass


def get_neighbouring_chunks(session: Session, chunk: Chunk) -> Tuple[Chunk | None, Chunk | None]:
    """
    Returns a tuple with the previous and the following `Chunk` 
    given an initial node characterised by a `filename` and a `chunk_id`
    """
    pass


def get_mentioned_entities(session: Session, chunk: Chunk, n_hops: int=1) -> dict:
    """ 
    Follows the `MENTIONS` relationships of a given Chunk in the Graph and collects mentioned entities. 
    `n_hops` is used to indicate the number of relationship layers that could be done following entities linking.  
    """
    pass


def filter_graph_by_communities(session: Session, community_ids: List[int], community_type: str="leiden"):
    """
    Creates a temporary  view of the Knowledge Graph to filter it into subgraphs given community ids.
    """
    pass