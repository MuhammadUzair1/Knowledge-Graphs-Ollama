from langchain.prompts import PromptTemplate

def get_rephrase_prompt() -> PromptTemplate:

    prompt = """
        Your task is to rephrase a user's question based on the schema of a Graph Database that will be given to you. 

        Do not mention anything else, just rephrase the question from the user to be as coherent as possible with the schema of the graph.
        Do not make things up or add any information on your own. 

        SCHEMA: {schema}
        QUESTION: {question}

        REPHRASED_QUESTION: 
    """

    template = PromptTemplate.from_template(prompt)

    template.input_variables = ['schema', 'question']

    return template


def get_summarization_prompt() -> PromptTemplate:

    prompt = """
        Your task is to synthetize a clear and helpful answer to a question.

        The information to use for the task come from a Vector Database and from a Graph Database.
        
        In your task, you MUST use either the context obtained from a vector search on the Vector Database 
        and the query results given running a Cypher Query on the Graph Database. I

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