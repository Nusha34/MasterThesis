from build_graph import Builder
import pandas as pd
from gensim.models.poincare import PoincareModel

class Trainer:
    def __init__(self, path_relationship: str, path_concept: str):
        self.path_relationship=path_relationship
        self.path_concept=path_concept

    def trainer(self, epochs):
        relations=Builder(self.path_relationship, self.path_concept)
        short_rel_list=list(relations())[0:1000]
        model = PoincareModel(short_rel_list, size=2, burn_in=10)
        model.train(epochs=epochs)
        model.save('test_model')
        return model
        
