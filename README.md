# Title: "Supporting semantic mapping towards clinical terminologies with deep learning."

## Prerequisites
This project utilizes Docker for providing an OS-independent development and integration experience. We highly recommend using Visual Studio Code and the associated "Development Container" which allows direct access to an environment and shell with pre-installed Python, corresponding packages, and a specialized IDE experience. However, running Docker standalone is also possible by using the `Dockerfile` in the `devcontainer` folder. 

## Directory Structure

```
|- biobert/ (Experiments with biobert)
|- combination_three_methods/ (this folder contains a jupiter notebook with the pipeline to combine results from TF-IDF, Poincare and DeepWalk)
|- experimenting_embedding_fusion/  (this folder contains experiments of fusion during training of both hierarchical and semantical embeddings)
|- mapping/ (Mapping between semantical embedding and hierarchical embedding)
    |- bilstm/ (Experiments with BiLSTM model)
        |- deepwalk/
        |- poincare/
    |- transformer/ (Experiments with Transformer models)
        |- deepwalk/
        |- poincare/
|- hierachical_structure/ (Experiments of learning the hierarchical structure of the SNOMED CT within OMOP)
    |- deepwalk/
    |- node2vec/
    |- poincare/
|- word2vec/ (Experiments of learning semantical relationships of English medical-related data)
    |- pubmed/
    |- wiki/ 
```