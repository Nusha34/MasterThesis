import pandas as pd


class Builder:
    def __init__(self, path_relationship: str, path_concept: str):
        self.path_relationship=path_relationship
        self.path_concept=path_concept
        self.data_relationship=pd.read_csv(path_relationship, sep='\t')
        self.data_concept = pd.read_csv(path_concept, sep='\t')

    def __call__(self):
        data_relationship=self.data_relationship[self.data_relationship.relationship_id=='Is a']
        data_merge=data_relationship.merge(self.data_concept, left_on='concept_id_1', right_on='concept_id', how='left')
        data_merge_2=data_merge.merge(self.data_concept, left_on='concept_id_2', right_on='concept_id', how='left')      
        data_with_relationships=data_merge_2[['concept_id_1', 'concept_name_x', 'concept_id_2', 'concept_name_y', 'relationship_id']]
        for row_id, row in data_with_relationships.iterrows():
            yield (row['concept_id_1'], row['concept_id_2'])

    def get_dataframe(self):
        data_relationship=self.data_relationship[self.data_relationship.relationship_id=='Is a']
        data_merge=data_relationship.merge(self.data_concept, left_on='concept_id_1', right_on='concept_id', how='left')
        data_merge_2=data_merge.merge(self.data_concept, left_on='concept_id_2', right_on='concept_id', how='left')      
        data_with_relationships=data_merge_2[['concept_id_1', 'concept_name_x', 'concept_id_2', 'concept_name_y', 'relationship_id']]
        return data_with_relationships

