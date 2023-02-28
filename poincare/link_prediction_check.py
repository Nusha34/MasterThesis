from gensim.models.poincare import PoincareModel
from gensim.models.poincare import LinkPredictionEvaluation
import pandas as pd
import logging


logging.basicConfig(format="[%(asctime)s] %(message)s", level=logging.INFO)
logging.info("Load Model")
model=PoincareModel.load('/workspaces/master_thesis/results/puancare/model_20d')
logging.info("Load Datasets")
result=LinkPredictionEvaluation('/workspaces/master_thesis/rlationships_train.csv','/workspaces/master_thesis/rlationships_test.csv', model.kv)
logging.info("Evaluation")
evaluation=result.evaluate()
print(evaluation)
df_eval=pd.DataFrame(evaluation, index=[0])
df_eval.to_csv('linkpred_20d_.csv', index=False)

