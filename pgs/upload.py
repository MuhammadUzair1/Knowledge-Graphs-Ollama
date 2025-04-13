import os
import shutil
import streamlit as st

from src.config import Configuration
from src.graph.knowledge_graph import KnowledgeGraph
from src.ingestion.local_ingestor import LocalIngestor
from src.ingestion.chunker import Chunker
from src.ingestion.cleaner import Cleaner
from src.ingestion.embedder import ChunkEmbedder
from src.ingestion.graph_miner import GraphMiner

from pgs.utils import get_configuration_from_env

st.set_page_config(
    page_title="Upload",
    page_icon="🗳️",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    ## 🗳️ Ingestion of Files in the Graph 🕸️ 
    
    Use the box below to upload Files in `.pdf`, `.docx`, `.txt` or `.html` format.  
    They will be uploaded inside this App's root directory Source Folder and will then be available 
    for the ingestion process into your Knowledge Graph.  
    
    Each uploaded file will be passed into a pipeline that will; 
    * load it into a json format;
    * cleaning its text;
    * divide its text into smaller pieces, called chunks;
    * embed each chunk into its vector representation;
    * use a LLM model to extract a graph of concepts from each chunk;
    * upload the obtained vectors and entities into the Knowledge Graph;
    * update the centralities measures and the division of the Graph into communities.
    """
)


SOURCE_FOLDER = f"{os.getcwd()}/source_docs"
CONF_PATH = f"{os.getcwd()}/configuration.json"

env = False
conf = None
uploaded_files = None
st.session_state['ingest_clicked'] = False
st.session_state["cleanup_clicked"] = False
st.session_state["index_created"] = False


try:
    conf = Configuration.from_file(CONF_PATH)
except Exception as e:
    conf = get_configuration_from_env()
    

if conf or env:
    uploaded_files = st.file_uploader(
        label="Upload Files to ingest", 
        accept_multiple_files=True
    )

if len(uploaded_files) > 0:

    os.makedirs(SOURCE_FOLDER, exist_ok=True)

    for file in uploaded_files:
        file_name = file.name

        file_path = SOURCE_FOLDER + f"/{file_name}"

        with open(file_path, 'wb') as f:
            f.write(file.getbuffer())

    st.session_state["ingest_clicked"] = st.button(
        label="Ingest into Knowledge Graph",
        icon="🗳️"
    )

    if st.session_state["ingest_clicked"]: 
        with st.status(
            f"Ingesting {len(uploaded_files)} Documents...", 
            expanded=True
            ) as status:

            st.write("Setting up the Ingestion Pipeline ..")

            ingestor = LocalIngestor(source=conf.source_conf)
            cleaner = Cleaner()
            chunker = Chunker(conf=conf.chunker_conf)
            embedder = ChunkEmbedder(conf=conf.embedder_conf)
            graph_miner = GraphMiner(
                conf=conf.re_model_conf, 
                ontology=conf.database.ontology
            )
            knowledge_graph = KnowledgeGraph(
                conf=conf.database, 
                embeddings_model=embedder.embeddings,
            )
            if not knowledge_graph._driver.verify_authentication():
                st.error("Check your Neo4j Configuration!")
            
            # # TODO 
            # if not st.session_state["index_created"]:
            #     if not knowledge_graph.index_exists():
            #         st.session_state["index_created"] = knowledge_graph.create_index()
            
            else:
                st.write("Loading..")
                docs = ingestor.batch_ingest()

                st.write("Cleaning..")
                docs = cleaner.clean_documents(docs)

                st.write("Chunking..")
                docs = chunker.chunk_documents(docs)

                st.write("Embedding..")
                docs = embedder.embed_documents_chunks(docs)

                st.write("Extracting a Knowledge Graph from each file..")
                docs = graph_miner.mine_graph_from_docs(docs=docs)

                st.write("Uploading Data to Knowledge Graph..")
                knowledge_graph.add_documents(docs)
                
                st.write("Updating Communities and computing Centralities in the Graph..")
                knowledge_graph.update_centralities_and_communities()
                
                status.update(
                    label="Done with the Ingestion", 
                    state="complete", 
                    expanded=False
                )

        if status._current_state == "complete":
            st.success(body=f"Done with the Ingestion of {len(docs)} Files")

            st.session_state["cleanup_clicked"] = st.button(
                label="Cleanup Folder",
                help=f"Clicking this will delete files in folder {SOURCE_FOLDER}",
                icon="🗑️"
            )

        if st.session_state["cleanup_clicked"]:
            shutil.rmtree(SOURCE_FOLDER)
            st.info("Cleanup Completed!")
            