## Transfer learning based approaches
Transfer learning based approaches usually involve a general pre-training phase. For VGG this involves learning to classify a very broad set of pictures. For text data this usually involves next sentence prediction and masked language modeling.
A pre-trained model can then be fine-tuned to a more specific context, such as sentiment. This also requires additional training on (usually) large datasets. We can then either use the fine-tuned model (if our task is identical) or further tune it using our own data.

#### Steps:

1. **Load Libraries**: Import the necessary libraries for data loading, preprocessing, model building, and evaluation.
2. **Load the Data**: Load the dataset.
3. **Load the Model**: Initialize the pre-trained model.
4. **Split the Data**: Split the data into training and testing sets.
5. **Fit the Model**: Further train the model on the training data.
6. **Calculate the Sentiment**: Predict sentiments on the test data using the trained model.
7. **Evaluate the Performance**: Evaluate the model's performance using a classification report.

The `main.py` contains an example from start to finish using the Ikea dataset.