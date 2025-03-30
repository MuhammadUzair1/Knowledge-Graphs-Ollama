from typing import Optional, Any, Dict

from langchain_core.messages import BaseMessage
from langchain_neo4j.chains.graph_qa.cypher import GraphCypherQAChain

from src.config import LLMConf
from src.graph.knowledge_graph import KnowledgeGraph
from src.factory.llm import fetch_llm
from src.prompts.graph_qa import get_rephrase_prompt, get_summarization_prompt
from src.utils.logger import get_logger


logger = get_logger(__name__)


class GraphAgentResponder:
    """
    Agent powered by up to three LLMs, is able to answer a user's question
    navigating the `KnowledgeGraph` via Cypher Queries as well as via Vector Search.
    """

    def __init__(
        self, 
        qa_llm_conf: LLMConf,
        cypher_llm_conf: LLMConf, 
        graph: KnowledgeGraph,
        rephrase_llm_conf: Optional[LLMConf]=None
    ):
        self.graph = graph
        self.qa_llm = fetch_llm(qa_llm_conf)
        self.cypher_llm = fetch_llm(cypher_llm_conf)

        self.summarize_prompt = get_summarization_prompt()

        self.graph_qa_chain = GraphCypherQAChain.from_llm(
            qa_llm=self.qa_llm, 
            cypher_llm=self.cypher_llm,
            graph=self.graph, 
            verbose=True,
            allow_dangerous_requests=True,
            validate_cypher=True, 
            return_intermediate_steps=True
        )
        self.rephrase_llm = None
        if rephrase_llm_conf:
            self.rephrase_llm = fetch_llm(rephrase_llm_conf)
            self.rephrase_prompt = get_rephrase_prompt()
            self.rephrase_prompt.partial_variables = {
                "graph_labels": self.graph.labels,
                "graph_relationships": self.graph.relationships
            }


    def answer(self, query: str, filter:Optional[Dict[str, Any]]=None) -> str:
        """ 
        Asnwers the user query performing text generation after having retrieved
        context both via Vector Search and Cypher Queries. 
        Results from both this methods are synthetized in a comprehensive answer.

        If a configuration is provided for the rephrasing LLM, it will be used 
        to rephrase the user's query according to the `KnowledgeGraph` schema. 
        """
        context = ""
        
        try:
            context_docs = self.graph.vector_store.similarity_search(
                query=query,
                k=5,
                filter=filter if filter else None
            )
        except Exception as e:
            logger.warning(f"Failed to retrieve context with exception: {e}")
            context_docs = []

        for doc in context_docs:
            context += f"\n {doc.page_content}"

        if self.rephrase_llm:
            try: 
                rephrased_question = self.rephrase_llm.invoke(
                    input=self.rephrase_prompt.format(
                        question=query)
                    ).content
            except Exception as e:
                logger.warning(f"Failed to rephrase user question with exception: {e}")
                rephrased_question = None
        else:
            rephrased_question = None

        try:
            graph_qa_output = self.graph_qa_chain.invoke(
                rephrased_question if rephrased_question else query
            )
        except Exception as e:
            graph_qa_output = None

        final_answer: BaseMessage = self.qa_llm.invoke(
            input=self.summarize_prompt.format(
                question=query, 
                retrieved_context=context, 
                query_result=graph_qa_output['intermediate_steps'] if graph_qa_output else {}
            )
        )

        return final_answer.content
        