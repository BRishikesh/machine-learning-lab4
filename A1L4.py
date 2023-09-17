import pandas as pd
import math

# Load the dataset
data = {
    'age': ['<=30', '<=30', '31-40', '>40', '>40', '>40', '31-40', '<=30', '<=30', '>40', '<=30', '31-40', '31-40', '>40'],
    'income': ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium', 'low', 'medium', 'medium', 'medium', 'high', 'medium'],
    'student': ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no'],
    'credit_rating': ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'excellent'],
    'buys_computer': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
}

df = pd.DataFrame(data)

# Define a function to calculate entropy
def entropy(column):
    value_counts = column.value_counts()
    total = len(column)
    entropy = -sum((count / total) * math.log2(count / total) for count in value_counts)
    return entropy

# Define a function to calculate information gain
def information_gain(data, target_column, feature_column):
    total_entropy = entropy(data[target_column])
    weighted_entropy = 0
    unique_values = data[feature_column].unique()
    
    for value in unique_values:
        subset = data[data[feature_column] == value]
        weighted_entropy += (len(subset) / len(data)) * entropy(subset[target_column])
    
    return total_entropy - weighted_entropy

# Calculate Information Gain for each attribute
attributes = ['age', 'income', 'student', 'credit_rating']
info_gain_dict = {}

for attribute in attributes:
    info_gain = information_gain(df, 'buys_computer', attribute)
    info_gain_dict[attribute] = info_gain

# Find the feature with the highest Information Gain (root node)
root_feature = max(info_gain_dict, key=info_gain_dict.get)

print("Information Gain for each attribute:")
print(info_gain_dict)
print("The first feature for constructing the decision tree (root node) is:", root_feature)
