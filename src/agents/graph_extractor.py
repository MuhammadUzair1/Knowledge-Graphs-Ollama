from logging import getLogger
from typing import Optional

from langchain_neo4j.graphs.graph_document import GraphDocument

from src.agents.llm import fetch_llm
from src.config import LLMConf
from src.schema import Ontology
from src.prompts.graph_extractor import get_graph_extractor_prompt 


logger = getLogger(__name__)


class GraphExtractor:
    """ Agent able to extract informations in a graph representation format from a given text.
    """

    def __init__(self, conf: LLMConf, ontology: Optional[Ontology]):
        self.conf = conf
        self.llm = fetch_llm(conf)
        self.prompt = get_graph_extractor_prompt()

        self.prompt.partial_variables = {
            'allowed_labels':ontology.allowed_labels, 
            'labels_descriptions': ontology.labels_descriptions, 
            'allowed_relationships': ontology.allowed_relations
        }


    def extract_graph(self, text: str) -> GraphDocument:
        """ Extracts a graph from a text.
        """
        if self.llm is not None:
            try:
                output: GraphDocument = self.llm.with_structured_output(
                    schema=GraphDocument
                    ).invoke(
                        input=self.prompt.format(input_text=text)
                    )
                
                return output
            except Exception as e:
                logger.warning(f"Error while extracting graph: {e}")