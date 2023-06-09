import logging
import random
import pandas as pd
from gensim.models import Word2Vec
import networkx as nx
import sys
sys.path.append('/workspaces/master_thesis/poincare')
from build_graph import Builder


logging.basicConfig(format="[%(asctime)s] %(message)s", level=logging.INFO)


def get_random_walk(graph: nx.Graph, node: int, n_steps: int = 20):
    """ Given a graph and a node, 
        return a random walk starting from the node 
    """
    local_path = [str(node), ]
    target_node = node
    for _ in range(n_steps):
        neighbors = list(nx.all_neighbors(graph, target_node))
        target_node = random.choice(neighbors)
        local_path.append(str(target_node))
    return local_path


relations = Builder('/workspaces/master_thesis/CONCEPT_RELATIONSHIP.csv',
                    '/workspaces/master_thesis/CONCEPT.csv')
relations = list(relations())
G = nx.Graph()
G.add_edges_from(relations)
print('Graph built')


walk_paths = []
for node in G.nodes():
    for _ in range(10):
        walk_paths.append(get_random_walk(G, node))

print('Walk paths built')
print('Training model...')
# Instantiate word2vec model
embedder = Word2Vec(
    window=20, sg=1, hs=0, negative=10, alpha=0.03, min_alpha=0.0001,
    seed=42
)
# Build Vocabulary
embedder.build_vocab(walk_paths, progress_per=2)
# Train
embedder.train(
    walk_paths, total_examples=embedder.corpus_count, epochs=20,
    report_delay=1
)
print('Model trained')
embedder.save('/workspaces/master_thesis/deepwalk_snomed.model')
