from logging import getLogger
from typing import List

from langchain_neo4j.graphs.graph_document import GraphDocument

from src.agents.graph_extractor import GraphExtractor
from src.config import LLMConf
from src.schema import Ontology, ProcessedDocument, Chunk

logger = getLogger(__name__)


class GraphMiner:
    """ Contains methods to mine graphs from a (list of) `ProcessedDocument`."""

    def __init__(self, conf: LLMConf, ontology: Ontology):
        self.graph_extractor = GraphExtractor(conf=conf, ontology=ontology)

        if self.graph_extractor:
            logger.info(f"GraphMiner initialized.")


    def mine_graph_from_doc(self, doc: ProcessedDocument) -> ProcessedDocument:
        """
        Mines a graph from a `ProcessedDocument` instance.
        """
        graph_docs = []
        for chunk in doc.chunks:
            try:
                graph: GraphDocument = self.graph_extractor.extract_graph(chunk.text)
                graph_docs.append(graph)
            
            except Exception as e:
                logger.warning(f"Error while mining graph: {e}")

        if len(graph_docs) > 0:
            doc.nodes = []
            doc.relationships = []

            for graph in graph_docs:
                doc.nodes.extend(graph.nodes)
                doc.relationships.extend(graph.relationships)

            logger.info(f"Created a graph representation with {len(doc.nodes)} Nodes and {len(doc.relationships)} Relations.")
        
        return doc


    def mine_graph_from_docs(self, docs: List[ProcessedDocument]) -> List[ProcessedDocument]:
        """
        Mines graphs from a list of `ProcessedDocument` instances.
        """
        return [self.mine_graph_from_doc(doc) for doc in docs]
    