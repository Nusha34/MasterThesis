# Title: "Supporting semantic mapping towards clinical terminologies with deep learning."
## Abstract 

Standardizing medical data from diverse sources in healthcare research is a critical step for consistent analysis. While methods exist to bridge the gap between raw data and standardization, many overlook the hierarchical structure of vocabularies like SNOMED CT. This research aimed to use this structure in combination with semantic information to improve mapping between data and standard vocabularies. To achieve this, various methods were assessed, with the combination of Bi-LSTM and DeepWalk standing out, achieving an accuracy rate of 13.73% for k-1 nearest neighbours and approximately 40% for k-5. The combination of Bi-LSTM and Poincaré yielded a 20% accuracy for k-5. Consistently, Word2Vec was used for semantic embedding across all configurations. Interestingly, the traditional USAGI method outperformed our advanced approach. Nevertheless, when combined with our methods, the accuracy of mapping medical phrases to standard vocabulary improved significantly. This study underscores the efficiency of simpler methods like TF-IDF and showcases the potential of hybrid approaches in medical informatics.

## Prerequisites
This project utilizes Docker for providing an OS-independent development and integration experience. We highly recommend using Visual Studio Code and the associated "Development Container" which allows direct access to an environment and shell with pre-installed Python, corresponding packages, and a specialized IDE experience. However, running Docker standalone is also possible by using the `Dockerfile` in the `devcontainer` folder. 

## Description
During the course of our study, we conducted several specific experiments, each with a unique focus:

1. We performed [Experiments](hierachical_embeddings) to learn the hierarchical relationships in SNOMED CT and evaluate their effectiveness.
2. We endeavoured to [Capture](semantical_embeddings) semantic relationships by training a Word2Vec model on PubMed and Wiki corpora and then put it to the test on analogy and similarity/relatedness tasks.
3. We executed a [Mapping](mapping_between_sem_hierachy) task between semantic and hierarchical spaces using neural networks, such as Bi-LSTM and Transformers.

Apart from these main experiments, we also [Evaluated](mapping_between_sem_hierachy/test_on_hchs_data.ipynb) our overall method on a selection of 19 medical phrases from the HCHS dataset. The results were then [Combined](combination_three_methods/combination_of_results.ipynb) using the strategy detailed in the report.

In addition to the core experiments, we conducted several additional tests not directly documented in the thesis but are nonetheless available in this Repository.


## Below are the instructions for running the Jupyter Notebooks containing the final results:

These steps allow you to execute the [Evaluation](mapping_between_sem_hierachy/test_on_hchs_data.ipynb) on samples from the HCHS data, and the [Combination](combination_three_methods/combination_of_results.ipynb) of the three methods:

1. Download the Standard Vocabulary from the [Athena website](https://athena.ohdsi.org/search-terms/start) and adjust the file paths in Jupyter Notebooks accordingly.
2. Obtain the folder containing the models used in the final evaluation. This folder will be attached to the submitted Master Thesis.
3. Update the paths to these models in the notebooks.
4. Run the notebooks.



## Directory Structure

```
|- combination_three_methods/ (This folder contains a Jupiter notebook with the pipeline to combine results from TF-IDF, Poincare and DeepWalk)
|- experimenting_embedding_fusion/  (This folder contains experiments of fusion during training of both hierarchical and semantical embeddings. We did not use it in the final report)
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
    |- biobert/ (Experiments with biobert. We did not use it in the final version of the report)
```
