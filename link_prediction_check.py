from gensim.models.poincare import PoincareModel
from gensim.models.poincare import LinkPredictionEvaluation
import pandas as pd




model=PoincareModel.load('model_100d')
result=LinkPredictionEvaluation('/workspaces/master_thesis/rlationships_train.csv','/workspaces/master_thesis/rlationships_test.csv', model.kv)
print('I am  ready to evaluat')
evaluation=result.evaluate()
print(evaluation)
df_eval=pd.DataFrame(evaluation, index=[0])
df_eval.to_csv('linkpred_100d.csv', index=False)
