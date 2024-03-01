# Implemented models
Here we briefly desribe the models we used in our paper. We differentiate between dictionary-based, machine learning-bases, and transfer / few-shot learning-based approaches.

Below is a brief table with links to the respective sections and some examples implemented in this repository.

|Model | Concept | Example |
|---|---|---|
|[VADER](#valence-aware-dictionary-and-sentiment-reasoner-vader)| [Dictionary](#dictionary-approaches)|
|[Naive Bayes](#naive-bayes-nb) | [ML](#machine-learning-based)|
|[Logistic Regression](#logistic-regression) | [ML](#machine-learning-based)|
|[Support Vector Machines](#support-vector-machines) | [ML](#machine-learning-based)|
|[Decision Tree](#decision-tree-dt) | [ML](#machine-learning-based)|
|[Random Forest](#random-forest-rf) | [ML](#machine-learning-based)|
|[XGBoost](#xgboost) | [ML](#machine-learning-based)|
|[ANN](#ann) | [ML](#machine-learning-based)|
|[CNN](#xgboost) | [ML](#machine-learning-based)|
|[LSTM](#lstm) | [ML](#machine-learning-based)|
|[CNN-LSTM](#cnn-lstm) | [ML](#machine-learning-based)|
|[BERT](#bert-normal) | [Transfer](#transfer-learning-based)|
|[BERT (fine-tuned)](#bert-fine-tuned) | [Transfer](#transfer-learning-based)|
|[SieBERT](#siebert) | [Transfer](#transfer-learning-based)|
|[Few-Shot (GPT)](#few-shot-gpt) | [Few-Shot](#few-shot-learning-based)|
|[Few-Shot (SetFit)](#few-shot-sentence-transformer) | [Few-Shot](#few-shot-learning-based)|

## Dictionary Approaches
[Back to beginning](#implemented-models)

Dictionary approaches are caracterized by their ease of use, interpretability and simplicity. Researchers construct a collection of words and matching sentiment (or other linguistic concept) scores, which are used to rank a sentence. They are usually hard to construct and not easy to adjust.

### Valence Aware Dictionary and sEntiment Reasoner (VADER)
VADER uses lexicons and rules to assess 

See also their [github](https://github.com/cjhutto/vaderSentiment) repository.

## Machine Learning-Based
[Back to beginning](#implemented-models)

We consider both traditional approaches like Naive Bayes and more modern models, like CNN or LSTM under ML models. ML approaches can be caracterized by their flexibility and accuracy. 

### Naive Bayes (NB)
NB is a popular algorithm in a variety of classification tasks. Although more modern approaches oftentimes perform better, NB is a simple, fast and inexpensive benchmark. It works by calculating the conditional probability of a class given an attribute set.

If we wish to classify a text B using Naive Bayes, we use the formula:

```markdown
P(A|B) = [P(B|A) * P(A)] / P(B)
```
Here:
- A is the class or category we want to predict (e.g., positive or negative sentiment).
- B is the observed text.
- P(A|B) is the posterior probability of class A given predictor (text).
- P(B|A) is the likelihood which is the probability of predictor given class.
- P(A) is the prior probability of class.
- P(B) is the prior probability of predictor.

Due to its inexpensivenes, we can conduct very thorough HP training.

### Logistic Regression

### Support Vector Machines

### Decision Tree (DT)

### Random Forest (RF)

RF are an ensemble learning method and use multiple decision trees that are trained on subsets of the training data. These trees become "experts" and are pooled together for the final prediction. The pooling works by calculating the mode of the classes.

### XGBoost

### ANN

### CNN

### LSTM

### CNN-LSTM

## Transfer-Learning-Based
[Back to beginning](#implemented-models)

Under this section we consider normal pre-trained models like BERT but also versions fine-tuned on the SA task.

### BERT (normal)

### BERT (fine-tuned)

### SieBERT

## Few-Shot-Learning-Based
[Back to beginning](#implemented-models)

Under this section we consider normal pre-trained models like BERT but also versions fine-tuned on the SA task.
Few-Shot learning (FSL) involves providing very few examples (shots) to a model for it to learn the classification rules. In extreme examples this would be 0 (zero-shot), or 1 (one-shot). As a method it has seen wide application in computer vision.
In Natural Language Processing this is a rather recent option with the advent of transformers, GPT and sentence transformers.

### Few-Shot (GPT)
FSL using (Chat)GPT involves crafting a prompt which guides the model along the task we want to solve (few-shot-prompting).
An example prompt is given below
~~~python
prompt = """
    Sentence: I like this a lot.
    Sentiment: positive

    #####

    Sentence: this is very bad.
    Sentiment: negative

    #####

    Sentence: this was pretty good.
    Sentiment: positive

    #####

    Sentence: this could be worse.
    Sentiment: """
~~~
This prompt is then passed to (Chat)GPT with the "text completion task", i.e. fill out the missing information. Brown et al. (2020) have shown that GPT is a few-shot learner using this style of prompting.

We can then craft several payloads using known instances of our data and pass an unknown instance as the last sentence.

### Few-Shot (Sentence Transformer)