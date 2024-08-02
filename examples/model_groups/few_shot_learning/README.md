## Few-Shot learning based approaches
Few-Shot Learning (or sometimes k-shot learning) based approaches consist of using either specialized machine learning models or LLMs like ChatGPT to analyze and predict sentiments from text data.
The approach is different depending on the specific workflow but generally follows the below steps.

#### Steps:

1. **Load Libraries**: Import the necessary libraries for data loading, model building, and evaluation.
2. **Load the Data**: Load the dataset.
3. **Load the Model**: Initialize the model.
4. **Split the Data**: Split the data into training and testing sets.
5. **Subsample data**: Select k sentences per class for the k-shot learning.
6. **Fit the Model**: Train the model on the training data.
7. **Calculate the Sentiment**: Predict sentiments on the test data using the trained model.
8. **Evaluate the Performance**: Evaluate the model's performance using a classification report.

The `main.py` in either subdirectory contains an example from start to finish using the Ikea dataset.