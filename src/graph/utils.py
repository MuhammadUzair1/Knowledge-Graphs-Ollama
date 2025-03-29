import networkx as nx
from community import best_partition, modularity
from igraph import Graph
from leidenalg import find_partition, ModularityVertexPartition
from logging import getLogger

from typing import Tuple

logger = getLogger(__name__)


def detect_louvain_communities(G: nx.DiGraph, return_modularity:bool=True) -> nx.DiGraph | Tuple[nx.DiGraph, float]:
    """ 
    Detects Louvain communities for a `networkx` Directed Graph. 
    If `return_modularity`, also return the modularity of the Graph according to 
    the Louvain distance measure.
    """
    G_undirected = G.to_undirected()

    partition = best_partition(G_undirected)  # Louvain method

    nx.set_node_attributes(G, partition, "community_louvain")  # Store communities in node attributes

    if not return_modularity:

        return G
    
    else: 
        modularity = modularity(partition, G_undirected)

        logger.info(f"Modularity based on Louvain communities: {modularity}")

        return G, modularity 
    

def detect_leiden_communities(G: nx.DiGraph, return_modularity:bool=True) -> nx.DiGraph | Tuple[nx.DiGraph, float]:
    """
    Detects Leiden communities for a `networkx` Directed Graph. 
    If `return_modularity`, also return the modularity of the Graph according to 
    the Louvain distance measure.
    """
    
    # Convert networkx to igraph
    mapping = {node: i for i, node in enumerate(G.nodes())}  # Node mapping
    reverse_mapping = {i: node for node, i in mapping.items()}
    
    # Create igraph graph
    ig_G = Graph(directed=True)
    ig_G.add_vertices(len(G.nodes()))
    ig_G.add_edges([(mapping[u], mapping[v]) for u, v in G.edges()])
    
    partition = find_partition(ig_G, ModularityVertexPartition)

    # Assign community labels back to NetworkX
    for i, comm in enumerate(partition):
        for node in comm:
            G.nodes[reverse_mapping[node]]["community_leiden"] = i 
    
    if not return_modularity:
        return G
    
    else:
        modularity = partition.modularity

        logger.info(f"Modularity based on Leiden communities: {modularity}")

        return G, modularity
    

def compute_centralities(G: nx.DiGraph | nx.Graph) -> nx.DiGraph | nx.Graph:
    """
    Compute PageRank, Betweenness and Closeness Centralities and store them as metadata in the graph
    """
    
    pr = nx.pagerank(G, alpha=0.85)
    bc = nx.betweenness_centrality(G)
    cc = nx.closeness_centrality(G)

    nx.set_node_attributes(G, pr, "pagerank")
    nx.set_node_attributes(G, bc, "betweenness")
    nx.set_node_attributes(G, cc, "closeness")
    
    return G