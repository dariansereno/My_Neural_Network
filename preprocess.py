import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from tools import extract_csv

tolerance = 0.7

def preprocess(df: pd.DataFrame):
	df[0], df[1] = df[1].copy(), df[0].copy()
	le = LabelEncoder()
	df[0] = le.fit_transform(df[0])
	for i in range(31):
		if (i != 1):
			correlation = df[0].corr(df[i])
			if (correlation < tolerance):
				df.drop(i, axis=1, inplace=True)
	label_map = {old: new for new, old in enumerate(df.columns)}
	label_map[0] = "predict"
	df = df.rename(columns=label_map)
	data = df.drop("predict", axis=1)
	mean = np.mean(data, axis=0)
	std = np.std(data, axis=0)
	standardized_data = (data - mean) / std
	df[df.columns[1:]] = standardized_data

	return df


def split_dataset(df: pd.DataFrame):
	df = df.sample(frac=1).reset_index(drop=True)
	
	split_percentage = 0.8
	index = int(len(df) * split_percentage)
	Y = df["predict"]
	df = df.drop("predict", axis=1)

	return ((df.iloc[:index], df.iloc[index:]), (Y.iloc[:index], Y.iloc[index:]))


def get_data(df: pd.DataFrame):
	df = preprocess(df)
	return split_dataset(df)

df = extract_csv("data.csv")
df = preprocess(df)
(train_X, test_X), (train_Y, test_Y) = split_dataset(df)
print(train_X,np.array(test_Y).reshape(-1, 1))