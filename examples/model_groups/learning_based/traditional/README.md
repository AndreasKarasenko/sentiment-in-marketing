## Traditional learning based approaches
Learning based approaches consist of using machine learning models to analyze and predict sentiments from text data. Although the models are very different most Scikit-Learn models follow the same steps.

#### Steps:

1. **Load Libraries**: Import the necessary libraries for data loading, preprocessing, model building, and evaluation.
2. **Load the Data**: Load the dataset.
3. **Load the Model**: Initialize the model.
4. **Split the Data**: Split the data into training and testing sets.
5. **Preprocess the Data**: Use TF-IDF vectorization (or another embedding approach) for text data and label encoding for the target variable.
6. **Fit the Model**: Train the model on the training data.
7. **Calculate the Sentiment**: Predict sentiments on the test data using the trained model.
8. **Evaluate the Performance**: Evaluate the model's performance using a classification report.

The `main.py` contains an example from start to finish using the Ikea dataset.
This example can be directly used for DecisionTrees, XGBoost, SVM, Random Forests, etc.