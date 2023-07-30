from gensim.models import KeyedVectors
from gensim import models
from gensim.models import Word2Vec
import numpy as np
from gensim.models.poincare import PoincareModel
import nltk
import logging
from nltk.tokenize import word_tokenize
import pickle 

nltk.download('punkt')

def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms

def combine_weighted_sum(embedding1, embedding2, weight1, weight2):
    return weight1 * embedding1 + weight2 * embedding2

def sentence_embedding(sentence, word2vec_model):
    words = sentence.split()
    word_vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
    
    if not word_vectors:
        return None
    
    return np.mean(word_vectors, axis=0)

if __name__ == "__main__":
    logging.basicConfig(format="[%(asctime)s] %(message)s", level=logging.INFO)
    logging.info("Load word2vec model")
    word2vec_model = Word2Vec.load("/workspaces/master_thesis/word2vec_wiki_snomed_preprocessed.model")
    logging.info("Load poincare model")
    poincare_model=PoincareModel.load('/workspaces/master_thesis/poincare_300d_preprocessed')
    normalized_poincare_embeddings = normalize_embeddings(poincare_model.kv.vectors)
    normalized_word2vec_embeddings = normalize_embeddings(word2vec_model.wv.vectors)
    logging.info("Load Snomed concepts")
    with open("/workspaces/master_thesis/snomed_preprocessed", "rb") as fp:   # Unpickling
        concepts = pickle.load(fp)
    logging.info("Make a list out of concepts")
    list_of_concepts = [' '.join(concept) for concept in concepts]
    logging.info("Create a sentence embedding out of concepts")
    concept_word2vec_embeddings = {}
    for concept in list_of_concepts:
        concept_word2vec_embeddings[concept] = sentence_embedding(concept, word2vec_model)
    combined_embeddings = {}
    weight1 = 0.5   
    weight2 = 0.5  
    logging.info("Creating an embedding")     
    for concept in list_of_concepts:
        if concept in poincare_model.kv and concept in concept_word2vec_embeddings:
            combined_embeddings[concept] = combine_weighted_sum(
                normalized_poincare_embeddings[poincare_model.kv.key_to_index[concept]],
                normalized_word2vec_embeddings[list(concept_word2vec_embeddings.keys()).index(concept)],
                weight1, weight2
            )
        elif concept in concept_word2vec_embeddings:
            combined_embeddings[concept] = concept_word2vec_embeddings[concept]
    print(combined_embeddings)
    with open("combined_corpus_embedding", "wb") as fp: 
        pickle.dump(combined_embeddings, fp)