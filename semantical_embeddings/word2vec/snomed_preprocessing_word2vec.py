import pandas as pd
from concept import Concept
import spacy
from nltk.stem import PorterStemmer
import pickle
nlp = spacy.load("en_core_web_sm")

def preprocessing(sample):
    sample = sample.lower()
    stemmer = PorterStemmer()
    token_list = []
    doc = nlp(sample)
    token_list = [stemmer.stem(token.text)
            for token in doc
            if not token.is_stop and not token.is_punct
        ]
    sentence = " ".join(token_list)
    return sentence 

if __name__ == "__main__":
    concepts = pd.read_csv('/workspaces/master_thesis/CONCEPT.csv', on_bad_lines="skip", delimiter="\t", low_memory=False)
    synonyms = pd.read_csv('/workspaces/master_thesis/CONCEPT_SYNONYM.csv', on_bad_lines="skip", delimiter="\t", low_memory=False)
    concepts=Concept.concatenate_concept_with_their_synonyms(concepts, synonyms, ['SNOMED'])
    preprocessed_text = []
    print('Preprocessing')
    for concept in concepts.names:
        # Cleaning the text
        text = preprocessing(concept)
        # Appending to the all text list
        preprocessed_text.append(text.split())
    with open("snomed_preprocessed", "wb") as fp:   
        pickle.dump(preprocessed_text, fp)