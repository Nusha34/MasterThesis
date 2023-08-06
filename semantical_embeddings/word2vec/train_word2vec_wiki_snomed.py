from gensim.models import Word2Vec
import pickle
import logging

logging.basicConfig(format="[%(asctime)s] %(message)s", level=logging.INFO)
logging.info("Loading trained wiki model")
model = Word2Vec.load("/workspaces/master_thesis/word2vec_wiki.model")
logging.info("Loading SNOMED corpus")
with open(
       "/workspaces/master_thesis/snomed_preprocessed", "rb"
   ) as fp:  # Unpickling
       snomed_corpus = pickle.load(fp)
logging.info("Training wiki+snomed")
model.train(snomed_corpus, total_examples=len(snomed_corpus), epochs=300)
logging.info("Training done.")
logging.info("Save model")
model.save("word2vec_wiki_snomed_preprocessed.model")