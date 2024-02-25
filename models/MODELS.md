# Implemented models
Here we briefly desribe the models we used in our paper. We differentiate between dictionary-based, machine learning-bases, and transfer / few-shot learning-based approaches.

Below is a brief table with links to the respective sections and some examples implemented in this repository.

|Model | Concept | Example |
|---|---|---|
|[LIWC](#liwc)| [Dictinoary](#dictionary-approaches)|
|[VADER](#valence-aware-dictionary-and-sentiment-reasoner-vader)| [Dictinoary](#dictionary-approaches)|
|[Naive Bayes](#naive-bayes-nb) | [ML](#machine-learning-based)|
|[Decision Tree](#decision-tree-dt) | [ML](#machine-learning-based)|
|[Random Forest](#random-forest-rf) | [ML](#machine-learning-based)|

## Dictionary Approaches
Dictionary approaches are caracterized by their ease of use, interpretability and simplicity. Researchers construct a collection of words and matching sentiment (or other linguistic concept) scores, which are used to rank a sentence. They are usually hard to construct and not easy to adjust.

### LIWC
### Valence Aware Dictionary and sEntiment Reasoner (VADER)
VADER uses lexicons and rules to assess 

See also their [github](https://github.com/cjhutto/vaderSentiment) repository.

## Machine Learning-Based
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

### Decision Tree (DT)

### Random Forest (RF)

RF are an ensemble learning method and use multiple decision trees that are trained on subsets of the training data. These trees become "experts" and are pooled together for the final prediction. The pooling works by calculating the mode of the classes.

### 