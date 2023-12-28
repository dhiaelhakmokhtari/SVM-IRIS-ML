import pandas as pd


def normalize(features):
    result = features.copy()
    for feature_name in features.columns:
        if feature_name != 'variety':
            max_value = features[feature_name].max()
            min_value = features[feature_name].min()
            result[feature_name] = (features[feature_name] - min_value) / (max_value - min_value)
    return result


df = pd.read_csv("Dataset/modified_iris.csv")

# Normalize the specified features
normalized_df = normalize(df)

# Save the normalized data to a new CSV file
normalized_df.to_csv('modified_iris_normalized.csv', index=False)
