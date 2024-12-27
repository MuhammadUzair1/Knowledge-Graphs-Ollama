# knowledge-graphs
Repo created to create and interact with knowledge graphs 

## Graph QA
A typical graph query to “Find the top 10 most cited articles” would look like this:

````
MATCH(n:Article) 
WHERE n.citation_count > 50
RETURN n.title, n.citation_count
````

## Vector Search:
“Find articles about climate change” would look like this in Semantic Retrieval:
````
query = "Find articles about climate change? "
vectorstore = Neo4jVector.from_existing_graph(**args)
vectorstore.similarity_search(query, k=3)
````


## Core GraphRAG Retrieval Pattern:
1. Do a vector search to find an initial set of nodes
2. Traverse the graph around those nodes to add context
3. (Optional:) Rank the results using the graph and pass the top-k documents to the LLM