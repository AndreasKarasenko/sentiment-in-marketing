# Sentiment Analysis in Marketing – From Fundamentals to State-of-the-art
This project serves as the online appendix for the paper ***Sentiment Analysis in Marketing – From Fundamentals to State-of-the-art***

The project structure is as follows:

```
config          --  config files (e.g. data config, or model config / hyperparameter specification)
data            --  folder where data should be stored
models          --  folder where models should be specified
ressources      --  folder for additional ressources
results         --  results for each model / dataset and aggregated
samples         --  train / test samples for reproducible testing
utils           --  utilities like eval functions
```

Everything is written as a Python Module so top Module like import is supported.

### Outline
The main goal is to provide an overview of relevant classical as well as more modern and state of the art (SOTA) approaches to text classification and explicitly sentiment analysis.

**General Problem**: Given a text *T* we want to predict *n* classes representing emotional valence. Since computers can't work with words we need to encode them to numbers which leads us to the first sub-topic.

**Preprocessing**: Various approaches to transform text exist. Among them are: BoW, AutoEncoders, Pre-Trained embeddings (Glove) and modern Pre-Trained embeddings from BERT. We show how to use each approach and provide comparisons regarding predictive accuracy.

**Training / Hyperparameter Tuning**: Since default values make little sense, HP tuning is a standard process. We show how to unify both traditional models like RF with more modern approaches like CNN in a single scikit-learn pipeline.

## Notes
For docstrings I used Github Copilot and adjusted where necessary.

Libraries like cuML promise speed ups by utilizing GPUs but also require more complicated setups. At the same time the scikit-learn implementations are by no means slow. If the setup proves to be problematic, we would recommend using the scikit-learn classes /functions instead. Thanks to the efforts of the cuML dev team, the code should be easily transferable.

For an installation guide of cuML go [here](./cuML_install.txt) or [here]()