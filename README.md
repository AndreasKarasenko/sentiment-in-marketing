# Sentiment Analysis in Marketing – From Fundamentals to State-of-the-art
This project serves as the online appendix for the paper ***Sentiment Analysis in Marketing – From Fundamentals to State-of-the-art***.

We compared several models from Naive Bayes to Few-Shot prompting with ChatGPT.
If you simply want to see results check out the [overall results](./results/overview/overall_scores_F1.xlsx),
or the [average results](./results/overview/summary_average.xlsx).
If you want to see some examples of how to work with text data see the [examples](./examples/).
We cover scraping, preprocessing, modeling, metrics, and interpretability.

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
We additionally have several top-level python scripts that ran both the hyperparameter estimation (where appropriate) and the evaluation of the models.
These are explained further down.

The requirements are in `reqs_setfit.txt` and `requirements.txt`. If you want to experiment with cuML (for GPU accelerated Scikit-Learn) you also need `cuML_install.txt`.


Everything is written as a Python Module so top Module like import is supported.

To replicate the results of the paper you have to run `eval_ml.py`, `eval_cnn.py`, `eval_transformer.py`, `eval_gpt.py`, and `eval_setfit.py`. We used `scrape_data.py` to scrape the Google Play Store datasets.
We then used `sample_data.py` to split the data into train and test sets **once** to ensure comparability between models.
We additionaly provide code examples for Neural Networks and LSTM (`run_dnn.py`, `run_lstm.py`).

For GPU accelerated evaluations you can try `eval_gpu_ml.py` for all Scikit-Learn models. 

Some subfolders like [utils](./utils/) provide additional information about their scripts.
The [examples](./examples/) folder contains various working examples for scraping, loading data, preprocessing, metrics, end-to-end examples for various models (Vader, Naive Bayes, CNN, transfer-learning, few-shot-learning), and also ways to interpret such black-box models.

The [data](./data/) folder contains all used data, and our train / test splits can be found in [samples](./samples/). 

The [config](./config/) folder contains model checkpoints, search spaces for Hyperparameter Tuning, data config files (for text and target columns) and command line utilities (see [argparse_args.py](./config/utils_config/argparse_args.py)).


## Notes

Libraries like cuML promise speed ups by utilizing GPUs but also require more complicated setups. At the same time the scikit-learn implementations are by no means slow. If the setup proves to be problematic, we would recommend using the scikit-learn classes /functions instead. Thanks to the efforts of the cuML dev team, the code should be easily transferable.

For an installation guide of cuML go [here](./cuML_install.txt).