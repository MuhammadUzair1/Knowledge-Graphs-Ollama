from langchain.prompts import PromptTemplate

def get_rephrase_prompt() -> PromptTemplate:

    prompt = """
        Your task is to rephrase a user's question based on the schema of a Graph Database that 
        will be given to you. Such schema is made of node labels and relationships available in the Graph.

        Remember that in a Knowledge Graph there are Documents and Chunks. 
        * a node with label `Document` always has a property `filename` (every Document has a name);
        * a node with label `Chunk` is connected via a `PART_OF` relationship to a node with the `Document` label (Chunks are pieces of text coming from a Document);
        * a node with label `Chunk` always has a `text` property; 
        * a node with label `Chunk` is usually connected to another node with label `Chunk` by a `NEXT` relationship (Chunks are ordered in a sequential order);
        * a node with label `Chunk` might be connected to other nodes in the Graph by a `MENTIONS` relationship (text in Chunks might mention some relevant entities). 

        Do not mention anything else, just rephrase the question from the user to be as coherent as possible with the schema of the graph.
        Do not make things up or add any information on your own. 

        AVAILABLE NODE LABELS: {graph_labels}
        AVAILABLE RELATIONSHIPS: {graph_relationships}
        QUESTION: {question}

        REPHRASED_QUESTION: 
    """

    template = PromptTemplate.from_template(prompt)

    template.input_variables = ['graph_labels', 'graph_relationships', 'question']

    return template


def get_summarization_prompt() -> PromptTemplate:

    prompt = """
        Your task is to synthetize a clear and helpful answer to a question.

        The suources of information to use for your task come from a Vector Database and from a Graph Database.
        
        In your task, you MUST use the context obtained from a vector search on the Vector Database 
        and the query results given running a Cypher Query on the Graph Database.  
        If one of the sources is empty, just answer the question using the other source. 

        Do not mention anything else, just summarize an precise, clear and helpful answer. 
        Do not make things up or add any information on your own. 

        QUESTION: {question}

        RETRIEVED CONTEXT: {retrieved_context}

        QUERY RESULT ON GRAPH: {query_result}

        ANSWER: 
    """

    template = PromptTemplate.from_template(prompt)

    template.input_variables = ['question', 'retrieved_context', 'query_result']

    return template