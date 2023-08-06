from puancare_train import Trainer
import logging

logging.basicConfig(format="[%(asctime)s] %(message)s", level=logging.INFO)
tr=Trainer('/workspaces/master_thesis/CONCEPT_RELATIONSHIP.csv', '/workspaces/master_thesis/CONCEPT.csv')
model_two=tr.trainer(100)