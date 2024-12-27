from typing import List

from langchain_core.prompts import PromptTemplate


def get_graph_extractor_prompt(
        # input_text: str,
        # allowed_labels: List[str]=None,
        # labels_descriptions: List[dict]=None,
        # allowed_relationships: List[str]=None
)-> PromptTemplate:
    """ 
    Parses the instructions to give as input to the LLM in charge of 
    Relations Extractions. 
    """
    prompt= """
        Knowledge Graph Instructions for LLM
        ## 1. Overview
        You are a top-tier algorithm designed for extracting information in structured formats to build a knowledge graph.
        - **Nodes** represent entities and concepts. They're akin to Wikipedia nodes.
        - The aim is to achieve simplicity and clarity in the knowledge graph, making it accessible for a vast audience.
        ## 2. Labeling Nodes
        - **Consistency**: Ensure you use basic or elementary types for node labels.
            - For example, when you identify an entity representing a person, always label it as **"person"**. Avoid using more specific terms like "mathematician" or "scientist".
        - **Node IDs**: Never utilize integers as node IDs. Node IDs should be names or human-readable identifiers found in the text.
        - **Allowed Node Labels:** {allowed_labels}
        - **Labels Descriptions:** {labels_descriptions}
        - **Allowed Relationship Types**: {allowed_relationships}
        ## 3. Handling Numerical Data and Dates
        - Numerical data, like age or other related information, should be incorporated as attributes or properties of the respective nodes.
        - **No Separate Nodes for Dates/Numbers**: Do not create separate nodes for dates or numerical values. Always attach them as attributes or properties of nodes.
        - **Property Format**: Properties must be in a key-value format.
        - **Quotation Marks**: Never use escaped single or double quotes within property values.
        - **Naming Convention**: Use camelCase for property keys, e.g., `birthDate`.
        ## 4. Coreference Resolution
        - **Maintain Entity Consistency**: When extracting entities, it's vital to ensure consistency.
        If an entity, such as "John Doe", is mentioned multiple times in the text but is referred to by different names or pronouns (e.g., "Joe", "he"),
        always use the most complete identifier for that entity throughout the knowledge graph. In this example, use "John Doe" as the entity ID.
        Remember, the knowledge graph should be coherent and easily understandable, so maintaining consistency in entity references is crucial.
        ## 5. Strict Compliance
        Adhere to the rules strictly. Non-compliance will result in termination.

        ## Begin Extraction!
        {input_text}
    """

    template = PromptTemplate.from_template(prompt)

    template.input_variables = ['input_text', 'allowed_labels', 'labels_descriptions', 'allowed_relationships']

    return template