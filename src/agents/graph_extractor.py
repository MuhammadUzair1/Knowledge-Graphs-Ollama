from logging import getLogger
from typing import Optional

# from langchain_neo4j.graphs.graph_document import Relationship, Node

from langchain.schema import Document

from src.agents.llm import fetch_llm
from src.config import LLMConf
from src.graph.graph_model import Ontology, _Graph, GraphDocument, graph_document_to_digraph
from src.prompts.graph_extractor import get_graph_extractor_prompt


logger = getLogger(__name__)


class GraphExtractor:
    """ Agent able to extract informations in a graph representation format from a given text.
    """

    def __init__(self, conf: LLMConf, ontology: Optional[Ontology]):
        self.conf = conf
        self.llm = fetch_llm(conf)
        self.allowed_labels = ontology.allowed_labels
        self.allowed_relationships = ontology.allowed_relations
        self.labels_descriptions = ontology.labels_descriptions
        self.prompt = get_graph_extractor_prompt()

        self.prompt.partial_variables = {
            'allowed_labels':ontology.allowed_labels if ontology.allowed_labels else "", 
            'labels_descriptions': ontology.labels_descriptions if ontology.labels_descriptions else "", 
            'allowed_relationships': ontology.allowed_relations if ontology.allowed_relations else ""
        }


    def extract_graph(self, text: str) -> _Graph:
        """ Extracts a graph from a text.
        """

        if self.llm is not None:
            try:
                graph: _Graph = self.llm.with_structured_output(
                    schema=_Graph
                    ).invoke(
                        input=self.prompt.format(input_text=text)
                    )

                return graph 
                
            except Exception as e:
                logger.warning(f"Error while extracting graph: {e}")