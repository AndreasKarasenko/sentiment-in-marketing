## Deep Learning based approaches
Deep learning based approaches use neural networks to analyze and predict sentiments from text data. Below is an example of using a Convolutional Neural Network (CNN) for sentiment analysis on a dataset of IKEA reviews.

### Example: Sentiment Analysis with CNN

This example demonstrates how to use a CNN to perform sentiment analysis on a dataset of IKEA reviews.

#### Steps:

1. **Load Libraries**: Import the necessary libraries for data loading, preprocessing, model building, and evaluation.
2. **Load the Data**: Load the dataset and take a subsample for faster processing.
3. **Split the Data**: Split the data into training and testing sets.
4. **Preprocess the Data**: 
   - Tokenize the text data and convert it to sequences of integers.
   - Pad the sequences to ensure they have the same length.
   - Encode the target variable.
5. **Build the Model**: 
   - Create a Sequential model.
   - Add an Embedding layer.
   - Add Conv1D and MaxPooling1D layers.
   - Add a GlobalMaxPooling1D layer.
   - Add Dense layers for the output.
6. **Compile the Model**: Compile the model with an optimizer, loss function, and metrics.
7. **Fit the Model**: Train the model on the training data.
8. **Evaluate the Model**: Predict sentiments on the test data using the trained model and evaluate its performance using a classification report.

The `main.py` contains an example from start to finish using the Ikea dataset.

The number of layers can be adjusted and do not have to conform to the presented model.