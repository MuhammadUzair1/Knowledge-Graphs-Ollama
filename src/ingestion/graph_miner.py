from logging import getLogger
from typing import List, Optional

from src.agents.graph_extractor import GraphExtractor, Graph
from src.config import LLMConf
from src.schema import Ontology, ProcessedDocument

logger = getLogger(__name__)


class GraphMiner:
    """ Contains methods to mine graphs from a (list of) `ProcessedDocument`."""

    def __init__(self, conf: LLMConf, ontology: Optional[Ontology]=None):
        self.graph_extractor = GraphExtractor(conf=conf, ontology=ontology)

        if self.graph_extractor:
            logger.info(f"GraphMiner initialized.")


    def mine_graph_from_doc(self, doc: ProcessedDocument) -> ProcessedDocument:
        """
        Mines a graph from a `ProcessedDocument` instance.
        """

        for chunk in doc.chunks:
            try:
                graph: Graph = self.graph_extractor.extract_graph(chunk.text)
                chunk.nodes = graph.nodes
                chunk.relationships = graph.relationships
            except Exception as e:
                logger.warning(f"Error while mining graph: {e}")

            logger.info(f"Created a graph representation for {len(doc.chunks)} chunks.")
        
        return doc


    def mine_graph_from_docs(self, docs: List[ProcessedDocument]) -> List[ProcessedDocument]:
        """
        Mines graphs from a list of `ProcessedDocument` instances.
        """
        return [self.mine_graph_from_doc(doc) for doc in docs]
    