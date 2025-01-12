from logging import getLogger
from typing import Optional, List

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_neo4j.graphs.graph_document import GraphDocument
from langchain_neo4j.graphs.neo4j_graph import Neo4jGraph
from langchain_neo4j.vectorstores.neo4j_vector import Neo4jVector
from neo4j import ManagedTransaction

from src.config import KnowledgeGraphConfig
from src.graph.graph_model import Ontology
from src.schema import ProcessedDocument


logger = getLogger(__name__)

BASE_ENTITY_LABEL = "__Entity__"


class KnowledgeGraph(Neo4jGraph):
    """
        Class used to represent a Knowledge Base under graph representation, 
        using `neo4j` as the backend for querying operations.
        
        If an `Ontology` is provided (see `KnowledgeGraphConfig.ontology`), will not allow for nodes and relationships
        to be created outside of the given sets of allowed labels and relationships.
    """

    def __init__(
            self, 
            conf: KnowledgeGraphConfig,
            embeddings_model: Embeddings,
            sanitize = False, 
            refresh_schema = True, 
            enhanced_schema = False
        ):
        if conf.uri is not None:
            self.url = conf.uri
        else: 
            self.url = f"{conf.db_schema}://{conf.host_name}:{conf.port}"
        self.username = conf.user
        self.password = conf.password
        self.database = conf.database
        self.timeout = conf.timeout
        self.index_name = conf.index_name

        if conf.ontology: # TODO 
            self.allowed_labels = conf.ontology.allowed_labels
            self.allowed_relationships = conf.ontology.allowed_relations

        self._labels_ = None 
        self._number_of_entities_ = None
        self._number_of_labels_ = None
        self._number_of_relationships_ = None
        self._number_of_docs = None
        self._relationships_ = None

        try: 
            self.vector_store = Neo4jVector(
                embedding=embeddings_model,
                url=self.url,
                username=self.username, 
                database=self.database,
                password=self.password,
                index_name=self.index_name,
                node_label="Chunk",
                embedding_node_property="embedding",
                text_node_property="text",
            )
        except Exception as e:
            logger.warning(f"Error connecting to Neo4jVector: {e}")

        super().__init__(
            url=self.url, 
            username=self.username,
            password=self.password,
            database=self.database,
            timeout=self.timeout,
            sanitize=sanitize, 
            refresh_schema=refresh_schema,
            enhanced_schema=enhanced_schema
        )

    @property
    def labels(self) -> List[str]:
        """
        Returns a list of labels in the Knowledge Graph.
        """
        with self._driver.session(database=self._database) as session:
            query = "CALL db.labels() YIELD label RETURN COLLECT(label) AS labels"
            result = session.run(query)
            self._labels = result.single()["labels"]
        return self._labels
    

    @property
    def relationships(self) -> List[str]:
        """
        Returns a list of relationships in the Knowledge Graph.
        """
        with self._driver.session(database=self._database) as session:
            query = "CALL db.relationshipTypes() YIELD relationshipType RETURN COLLECT(relationshipType) AS relationship_types"
            result = session.run(query)
            self._relationships_ = result.single()["relationship_types"]
        return self._relationships_
    

    @property
    def number_of_nodes(self) -> int:
        """
        Returns the total number of nodes in the Knowledge Graph.
        """
        with self._driver.session(database=self._database) as session:
            query = "MATCH (n) RETURN COUNT(n) AS nodes"
            result = session.run(query)
            self._number_of_entities = result.single()["nodes"]
        return self._number_of_entities
    

    @property
    def number_of_labels(self) -> int:
        """
        Returns the number of labels in the Knowledge Graph.
        """
        with self._driver.session(database=self._database) as session:
            query = "CALL db.labels() YIELD label RETURN COUNT(label) AS num_labels"
            result = session.run(query)
            self._number_of_labels = result.single()["num_labels"]
        return self._number_of_labels
    

    @property
    def number_of_relationships(self) -> int:
        """
        Returns the total number of relationships in the Knowledge Graph.
        """
        with self._driver.session(database=self._database) as session:
            query = "MATCH ()-[r]-() RETURN COUNT(r) AS num_relationships"
            result = session.run(query)
            self._number_of_relationships = result.single()["num_relationships"]
        return self._number_of_relationships
    

    @property
    def number_of_docs(self) -> int:
        """
        Returns the current number of documents collected in the Knowledge Graph
        """
        with self._driver.session(database=self._database) as session:
            query = "MATCH (n: Document) RETURN COUNT(n) AS num_docs"
            result = session.run(query)
            self.number_of_docs = result.single()["num_docs"]
        return self._number_of_docs
    

    @staticmethod
    def _create_document_node(tx: ManagedTransaction, doc: ProcessedDocument):
        query = """
            CREATE (d:Document {
                filename: $filename,
                document_version: $document_version,
                source: $source
            })
        """
        try:
            tx.run(
                query, 
                filename=doc.filename, 
                document_version=doc.document_version, 
                metadata=doc.metadata, 
                source=doc.source
            )
            logger.info(f"Document node created for file: {doc.filename}")
        except Exception as e:
            logger.warning(f"Error creating Document node for file: {doc.filename}")


    @staticmethod
    def _create_part_of_relationships(tx: ManagedTransaction, filename: str, document_version: int):
        query = """
            MATCH (d:Document {filename: $filename, document_version: $document_version})
            MATCH (c:Chunk {filename: $filename, document_version: $document_version})
            MERGE (c)-[:PART_OF]->(d)
        """
        try:
            tx.run(query, filename=filename, document_version=document_version)
            logger.info(f"PART_OF relationships created for Document {filename} version {document_version}")
        except Exception as e:
            logger.warning(f"Error creating PART_OF relationships for Document {filename}: {e}")
            

    @staticmethod
    def _create_next_relationships(
        tx: ManagedTransaction, 
        filename: str, 
        document_version: int
        ):
        query = """
            MATCH (c1:Chunk {filename: $filename, document_version: $document_version})
            WITH c1
            MATCH (c2:Chunk {filename: $filename, document_version: $document_version, chunk_id: c1.chunk_id + 1})
            MERGE (c1)-[:NEXT]->(c2)
        """
        try:
            tx.run(query, filename=filename, document_version=document_version)
        except Exception as e:
            logger.warning(f"Error creating NEXT relationships for chunks in Document {filename}: {e}")


    @staticmethod
    def _create_mentions_relationships(
        tx: ManagedTransaction, 
        node_id: str,
        chunk_id: int,
        filename: str,
        document_version: int
        ):
        query = """
            MATCH (c:Chunk {chunk_id: $chunk_id, filename: $filename, document_version: $document_version})
            MATCH (e:__Entity__ {id: $node_id})
            MERGE (c)-[:MENTIONS]->(e)
        """
        try:
            tx.run(
                query, 
                node_id=node_id, 
                chunk_id=chunk_id, 
                filename=filename, 
                document_version=document_version
            )
        except Exception as e:
            logger.warning(f"Error creating MENTIONS relationships for {node_id}: {e}")


    def create_document_node(self, doc: ProcessedDocument):
        """
        Creates a Document node in the Knowledge Graph.
        """
        with self._driver.session(database=self._database) as session:
            session.execute_write(
                self._create_document_node, 
                doc
            )
            session.execute_write(
                self._create_part_of_relationships, 
                doc.filename, 
                doc.document_version
            )
            logger.info(f"Document node created for file: {doc.filename}")

    
    def create_next_relationships(self, filename: str, doc_version: int):
        """
        Creates NEXT relationships between Chunk Nodes from a Document.
        """
        with self._driver.session(database=self._database) as session:
            session.execute_write(
                self._create_next_relationships, 
                filename,
                doc_version
            )
            logger.info(f"NEXT relationships created for Document {filename} version {doc_version}")


    def create_mentions_relationships(
            self, 
            node_id: str,
            chunk_id: int,
            filename: str,
            document_version: int
        ):
        """ Creates MENTIONS relationships between Chunk and __Entity__ nodes. """
        with self._driver.session(database=self._database) as session:
            session.execute_write(
                self._create_mentions_relationships, 
                node_id,
                chunk_id,
                filename,
                document_version
            )
            logger.info(f"MENTIONS relationships created!")


    def store_chunks_for_doc(self, doc: ProcessedDocument):
        """
        Stores Chunk nodes for a `ProcessedDocument` into the Knowledge Graph and updates the
        Knowledge Graph itself with the graphs extracted from each chunk, if any.
        """
        
        for chunk in doc.chunks:
            
            # doc level metadata
            if doc.metadata: 
                metadata = doc.metadata
            else: 
                metadata = {}
            metadata["filename"] = doc.filename
            metadata["document_version"] = doc.document_version
            # chunk level metadata
            metadata["chunk_id"] = chunk.chunk_id
            metadata["chunk_size"] = chunk.chunk_size
            metadata["chunk_overlap"] = chunk.chunk_overlap
            metadata["embeddings_model"] = chunk.embeddings_model

            try:
                self.vector_store.add_embeddings(
                    texts=[chunk.text],
                    embeddings=chunk.embedding,
                    metadatas=[metadata]
                )
            except Exception as e:
                logger.warning(f"Error storing chunk for document {doc.filename}: {e}")

            # store chunk's graph
            if chunk.nodes is not None :

                graph_doc: GraphDocument = GraphDocument(
                    nodes=chunk.nodes,
                    relationships=chunk.relationships if chunk.relationships is not None else [],
                    source=Document(
                        page_content=chunk.text,
                    )
                )

                try:
                    self.add_graph_documents(
                        graph_documents=[graph_doc], 
                        include_source=False,
                        baseEntityLabel=True
                    )

                    for node in chunk.nodes:
                        self.create_mentions_relationships(
                            node_id=node.id, 
                            chunk_id=chunk.chunk_id, 
                            filename=doc.filename, 
                            document_version=doc.document_version
                        )
                except Exception as e:
                    logger.warning(f"Error storing graph for chunk {chunk.chunk_id} in document {doc.filename}: {e}")

        try:
            self.create_next_relationships(
                filename=doc.filename,
                doc_version=doc.document_version
            )
        except Exception as e:
            logger.warning(f"Error creating NEXT relationships for chunks in Document {doc.filename}: {e}")

        try: 
            self.create_document_node(doc=doc)
        except Exception as e:
            logger.warning(f"Error creating Document source node for file: {doc.filename}: {e}")


    def add_documents(self, docs: List[ProcessedDocument]): 
        for doc in docs:
            self.store_chunks_for_doc(doc)
