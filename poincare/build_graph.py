import pandas as pd
import spacy
from nltk.stem import PorterStemmer


class Builder:
    def __init__(self, path_relationship: str, path_concept: str):
        self.path_relationship=path_relationship
        self.path_concept=path_concept
        self.data_relationship=pd.read_csv(path_relationship, on_bad_lines='skip', sep='\t')
        self.data_concept = pd.read_csv(path_concept, on_bad_lines='skip', sep='\t')
        self.nlp = spacy.load("en_core_web_sm")

    def __call__(self):
        data_relationship=self.data_relationship[self.data_relationship.relationship_id=='Is a']
        data_merge=data_relationship.merge(self.data_concept, left_on='concept_id_1', right_on='concept_id', how='left')
        data_merge_2=data_merge.merge(self.data_concept, left_on='concept_id_2', right_on='concept_id', how='left')   
        data_merge_2=data_merge_2[data_merge_2.standard_concept_x=='S'][data_merge_2.standard_concept_y=='S']
        data_with_relationships=data_merge_2[['concept_id_1', 'concept_name_x', 'concept_id_2', 'concept_name_y', 'relationship_id']]
        for row_id, row in data_with_relationships.iterrows():
            concept_x = self.preprocessing(row['concept_name_x'])
            concept_y = self.preprocessing(row['concept_name_y'])
            yield (concept_x, concept_y)

    def preprocessing(self, sample):
        sample = sample.lower()
        stemmer = PorterStemmer()
        token_list = []
        doc = self.nlp(sample)
        token_list = [stemmer.stem(token.text)
                for token in doc
                if not token.is_stop and not token.is_punct
            ]
        text = " ".join(token_list)
        return text  

    def get_dataframe(self):
        data_relationship=self.data_relationship[self.data_relationship.relationship_id=='Is a']
        data_merge=data_relationship.merge(self.data_concept, left_on='concept_id_1', right_on='concept_id', how='left')
        data_merge_2=data_merge.merge(self.data_concept, left_on='concept_id_2', right_on='concept_id', how='left')
        data_merge_2=data_merge_2[data_merge_2.standard_concept_x=='S'][data_merge_2.standard_concept_y=='S']      
        data_with_relationships=data_merge_2[['concept_id_1', 'concept_name_x', 'concept_id_2', 'concept_name_y', 'relationship_id']]
        return data_with_relationships

