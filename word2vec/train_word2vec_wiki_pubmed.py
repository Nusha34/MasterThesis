from gensim.models import Word2Vec
import pickle
import logging

logging.basicConfig(format="[%(asctime)s] %(message)s", level=logging.INFO)
logging.info("Loading trained wiki model")
model = Word2Vec.load("/workspaces/master_thesis/word2vec_wiki.model")
logging.info("Loading SNOMED corpus")
with open(
       "/workspaces/master_thesis/pubmed_corpus_full_300_1000", "rb"
   ) as fp:  # Unpickling
       pubmed_corpus = pickle.load(fp)
logging.info("Training wiki+pubmed")
model.train(pubmed_corpus, total_examples=1e6, epochs=300)
logging.info("Training done.")
logging.info("Save model")
model.save("word2vec_wiki_pubmed_1000.model")