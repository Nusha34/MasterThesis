from gensim.models.poincare import PoincareModel
from gensim.models.poincare import LinkPredictionEvaluation
import pandas as pd
import logging


logging.basicConfig(format="[%(asctime)s] %(message)s", level=logging.INFO)
logging.info("Load Model")
model=PoincareModel.load('/workspaces/master_thesis/model_400d_new')
logging.info("Load Datasets")
result=LinkPredictionEvaluation('/workspaces/master_thesis/poincare/relationships_train_1.csv','/workspaces/master_thesis/poincare/relationships_test_1.csv', model.kv)
logging.info("Evaluation")
evaluation=result.evaluate()
print(evaluation)
df_eval=pd.DataFrame(evaluation, index=[0])
df_eval.to_csv('linkpred_400d.csv', index=False)
logging.info("Done")
