
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = {
    'Age': ['30', '30', '35', '40', '40', '40', '36', '30', '30', '40', '30', '40', '38', '40'],
    'Income': ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium', 'low', 'medium', 'medium', 'medium', 'high', 'medium'],
    'Student': ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no'],
    'Credit_Rating': ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'excellent'],
    'Buys_Computer': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'no', 'no']
}


# Create a Decision Tree model
model = DecisionTreeClassifier()

df = pd.DataFrame(data)
Tr_X = df.drop(columns=['student'])
Tr_y = df['student'] 
# Fit the model to the training data
model.fit(Tr_X, Tr_y)

# Calculate the training set accuracy
training_accuracy = model.score(Tr_X, Tr_y)
print("Training Set Accuracy:", training_accuracy)

tree_depth = model.get_depth()
print("Depth of the Decision Tree:", tree_depth)
