import os

import streamlit as st

from src.config import Configuration
from src.agents.graph_qa import GraphAgentResponder
from src.graph.knowledge_graph import KnowledgeGraph
from src.ingestion.embedder import ChunkEmbedder

from pgs.utils import get_configuration_from_env

st.set_page_config(
    page_title="Chat",
    page_icon="🦜",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    ## Chat With Knowledge Graph

    After building a Knowledge Graph from your documents, you are now able to 
    ask it questions.  
    The agent in charge to answer ypu is able to both generate answers
    grounded by similarity search on document chunks, but also to query the Graph 
    in its own native language, Cypher. 
    """
)


CONF_PATH = f"{os.getcwd()}/configuration.json"

env = False
conf = None
if "messages" not in st.session_state:
    st.session_state.messages = []

try:
    conf = Configuration.from_file(CONF_PATH)
except Exception as e:
    conf = get_configuration_from_env()


if conf:
    embedder = ChunkEmbedder(conf=conf.embedder_conf)

    knowledge_graph = KnowledgeGraph(
        conf=conf.database, 
        embeddings_model=embedder.embeddings,
    )

    responder = GraphAgentResponder(
        qa_llm_conf=conf.qa_model,
        cypher_llm_conf=conf.qa_model,
        graph=knowledge_graph
        # rephrase_llm_conf=conf.qa_model
    )

    if knowledge_graph._driver.verify_authentication():
        
        a, b, c, d = st.columns(4, vertical_alignment="center")

        a.metric(
            label="# Docs in Graph",
            value=knowledge_graph.number_of_docs,
        )
        b.metric(
            label="# Labels in Graph",
            value=knowledge_graph.number_of_labels,
        )
        c.metric(
            label="# Nodes",
            value=knowledge_graph.number_of_nodes,
        )
        d.metric(
            label="# Relationships",
            value=knowledge_graph.number_of_relationships,
        )

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        with st.chat_message("assistant"):
            st.write("Hi, you can ask me questions about the Documents in the Knowledge Graph")
        
        # Accept user input
        if prompt := st.chat_input("What are the available nodes in the Graph?"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                # Add assistant response to chat history
                response =responder.answer(prompt)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

           