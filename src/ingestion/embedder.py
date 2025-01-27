from logging import getLogger
from typing import Union, List

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings, AzureOpenAIEmbeddings

from src.config import EmbedderConf
from src.schema import ProcessedDocument

logger = getLogger(__name__)


class ChunkEmbedder:
    """ Contains methods to embed Chunks from a (list of) `ProcessedDocument`."""
    def __init__(self, conf: EmbedderConf):
        self.conf = conf
        self.embeddings = self.get_embeddings()

        if self.embeddings:
            logger.info(f"Embedder of type '{self.conf.type}' initialized.")


    def get_embeddings(self) -> Union[HuggingFaceEmbeddings, OllamaEmbeddings, OpenAIEmbeddings, AzureOpenAIEmbeddings, None]:

        if self.conf.type == "ollama":
            embeddings = OllamaEmbeddings(
                model=self.conf.model
            )
        elif self.conf.type == "openai":
            embeddings = OpenAIEmbeddings(
                model=self.conf.model,
                api_key=self.conf.api_key,
                deployment=self.conf.deployment,
            )
        elif self.conf.type == "azure-openai":
            embeddings = AzureOpenAIEmbeddings(
                model=self.conf.model, 
                azure_endpoint=self.conf.endpoint,
                azure_deployment=self.conf.deployment,
                dimensions=1536,
                api_key=self.conf.api_key,
                api_version=self.conf.api_version
                
            )
        elif self.conf.type == "trf":
            embeddings = HuggingFaceEmbeddings(
                model=self.conf.model,
                endpoint=self.conf.endpoint,
            )
        else: 
            logger.warning(f"Embedder type '{self.conf.type}' not supported.")
            embeddings = None

        return embeddings
    

    def embed_document_chunks(self, doc: ProcessedDocument) -> ProcessedDocument:
        """
        Embeds the chunks of a `ProcessedDocument` instance.
        """
        if self.embeddings is not None:
            for chunk in doc.chunks:
                chunk.embedding = self.embeddings.embed_documents([chunk.text])
                chunk.embeddings_model = self.conf.model
            logger.info(f"Embedded {len(doc.chunks)} chunks.")
            return doc
        else: 
            logger.warning(f"Embedder type '{self.conf.type}' is not yet implemented")


    def embed_documents_chunks(self, docs: List[ProcessedDocument]) -> List[ProcessedDocument]:
        """
        Embeds the chunks of a list of `ProcessedDocument` instances.
        """
        if self.embeddings is not None:
            for doc in docs:
                doc = self.embed_document_chunks(doc)
            return docs
        else: 
            logger.warning(f"Embedder type '{self.conf.type}' is not yet implemented")
            return docs