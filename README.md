# Title: "Supporting semantic mapping towards clinical terminologies with deep learning."

## Prerequisites
This project utilizes Docker for providing an OS-independent development and integration experience. We highly recommend using Visual Studio Code and the associated "Development Container" which allows direct access to an environment and shell with pre-installed Python, corresponding packages, and a specialized IDE experience. However, running Docker standalone is also possible by using the `Dockerfile` in the `devcontainer` folder. 

## Description
We have conducted a series of different experiments in different directions: 
1. [Experiments](hierachical_embeddings) in learning hierarchical relationships of SNOMED CT and its evaluation;
2. [Capturing](semantical_embeddings) semantical relationships with Word2Vec embedding by training W2V on Pubmed and Wiki corpora and evaluate on analogy and similarity/relatedness tasks (semantical_embeddings/)
3. [Mapping](mapping_between_sem_hierachy) between semantical and hierarchical spaces by using neural networks: Bi-LSTM and Transformers

We [evaluated](mapping_between_sem_hierachy/test_on_hchs_data.ipynb) the whole approach on the 19 medical phrases from the HCHS dataset. And then [combined](combination_three_methods/combination_of_results.ipynb) them using the strategy described in the report. 

We also run additional experiments not described directly in the thesis but are also available in this Repository.


## Instruction on how to easily Run the Jupyter Notebooks with final results: 

The following instruction is to be able to run the [Evaluation](mapping_between_sem_hierachy/test_on_hchs_data.ipynb) on samples from HCHS data and the [Combination](combination_three_methods/combination_of_results.ipynb) of three approaches.
1. Download Standard Vocabulary from [Athena website](https://athena.ohdsi.org/search-terms/start) and change paths to files.
2. Get a folder with models used in the final evaluation, which will be attached for the Master Thesis
3. Change paths to these models in the notebooks 
4. Run



## Directory Structure

```
** |- combination_three_methods/ ** (This folder contains a Jupiter notebook with the pipeline to combine results from TF-IDF, Poincare and DeepWalk)
**|- experimenting_embedding_fusion/**  (This folder contains experiments of fusion during training of both hierarchical and semantical embeddings. We did not use it in the final report)
**|- mapping_between_sem_hierachy/** (Mapping between semantical embedding and hierarchical embedding)
    **|- bilstm/** (Experiments with BiLSTM model)
        **|- deepwalk/**
        **|- poincare/**
   **|- transformer/** (Experiments with Transformer models)
        **|- deepwalk/**
        **|- poincare/**
**|- hierachical_embeddings/** (Experiments of learning the hierarchical structure of the SNOMED CT within OMOP)
    **|- deepwalk/**
    **|- node2vec/**
    **|- poincare/**
**|- semantical_embeddings/** (Experiments of learning semantical relationships of English medical-related data)
    **|- word2vec/**   
        **|- pubmed/**
        **|- wiki/**
    **|- biobert/** (Experiments with biobert. We did not use it in the final version of the report)
```
