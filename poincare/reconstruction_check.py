from gensim.models.poincare import PoincareModel
from gensim.models.poincare import ReconstructionEvaluation
import pandas as pd
import logging


logging.basicConfig(format="[%(asctime)s] %(message)s", level=logging.INFO)
logging.info("Load Model")
model=PoincareModel.load('/workspaces/master_thesis/results/puancare/model_20d')
logging.info("Load Datasets")
result=ReconstructionEvaluation('/workspaces/master_thesis/poincare/relationships_train_1.csv', model.kv)
logging.info("Evaluation")
evaluation=result.evaluate()
print(evaluation)
df_eval=pd.DataFrame(evaluation, index=[0])
df_eval.to_csv('reconstruct_20d.csv', index=False)
logging.info("Done")
