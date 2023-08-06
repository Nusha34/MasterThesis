# Title: "Supporting semantic mapping towards clinical terminologies with deep learning."

## Prerequisites
This project utilizes Docker for providing an OS-independent development and integration experience. We highly recommend using Visual Studio Code and the associated "Development Container" which allows direct access to an environment and shell with pre-installed Python, corresponding packages, and a specialized IDE experience. However, running Docker standalone is also possible by using the `Dockerfile` in the `devcontainer` folder. 

## Description
We have conducted a series of different experiments in different directions: 
1. Experiments in learning hierachical relationships of SNOMED CT and its evaluation (hierachical_embeddings/);
2. Capturing semantical relationships with Word2Vec embedding, by training W2V on Pubmed and Wiki corpora and evaluate on analogy task and similarity/relatedness tasks (semantical_embeddings/)
3. Mapping between semantical and hierachical spaces by using neural networks: Bi-LSTM and Tranformers (mapping_between_sem_hierachy/)

We evaluated the whole approach on the 19 medical phrases from HCHS dataset ("master_thesis/mapping_between_sem_hierachy/test_on_hchs_data.ipynb"). And then combined them using the strategy based on the correlation between domain expert scores and similarity scores ("/master_thesis/combination_three_methods/combination_of_results.ipynb"). 

We also run additional experiments which are not described directly in the thesis but they are also available in this Repository.


## Instruction how to easily Run the Jupyter Notebooks with final results: 

The following instruction is to be able to run "master_thesis/mapping_between_sem_hierachy/test_on_hchs_data.ipynb" and "/workspaces/master_thesis/combination_three_methods/combination_of_results.ipynb"
1. Download Standard Vocabulary from [Athena website](https://athena.ohdsi.org/search-terms/start) and change paths to files.
2. Get folder with models used in the final evaluation which will be attached for the Master Thesis
3. Change paths to these models in the notebooks 
4. Run



## Directory Structure

```
|- combination_three_methods/ (This folder contains a jupiter notebook with the pipeline to combine results from TF-IDF, Poincare and DeepWalk)
|- experimenting_embedding_fusion/  (This folder contains experiments of fusion during training of both hierarchical and semantical embeddings. We did not use it the final report)
|- mapping_between_sem_hierachy/ (Mapping between semantical embedding and hierarchical embedding)
    |- bilstm/ (Experiments with BiLSTM model)
        |- deepwalk/
        |- poincare/
    |- transformer/ (Experiments with Transformer models)
        |- deepwalk/
        |- poincare/
|- hierachical_embeddings/ (Experiments of learning the hierarchical structure of the SNOMED CT within OMOP)
    |- deepwalk/
    |- node2vec/
    |- poincare/
|- semantical_embeddings/ (Experiments of learning semantical relationships of English medical-related data)
    |- word2vec/   
        |- pubmed/
        |- wiki/
    |- biobert/ (Experiments with biobert. We did not use it in final version of report)
```