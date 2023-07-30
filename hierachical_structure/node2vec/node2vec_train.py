import pandas as pd
import networkx as nx
from node2vec import Node2Vec


relationships = pd.read_csv('/workspaces/master_thesis/poincare/relationships_preprocessed.csv')
# relationships dataframe to list of tuples
relationships = [tuple(x) for x in relationships.values]
# relationships list of tuples to networkx graph
G = nx.Graph()
G.add_edges_from(relationships)
node2vec = Node2Vec(G, dimensions=128, walk_length=20, num_walks=5, workers=5)  # Use workers=1, quiet=True for less verbose output
model = node2vec.fit(window=10, min_count=1, batch_words=4)
# Save embeddings for later use
model.wv.save_word2vec_format('/workspaces/master_thesis/poincare/node2vec_embeddings')
#save model
model.save('/workspaces/master_thesis/poincare/node2vec_model')
