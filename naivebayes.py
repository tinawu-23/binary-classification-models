#!/usr/bin/env python3
import pandas as pd

# read training data
df = pd.read_csv('Dataset-football-train.txt', sep='\t', lineterminator='\n')
df_features = df[['Is_Home_or_Away','Is_Opponent_in_AP25_Preseason','Media']]

# P(Lose) P(lose)
l,w = df.groupby('Label').size()
pl = l / (l+w)
pw = w / (l+w)

# P(feature)
featuredict = {}
for feature in df_features:
	featuredict[feature] = df.groupby([feature,'Label']).size()

# read testing data
df_test = pd.read_csv('Dataset-football-test.txt', sep='\t', lineterminator='\n')
perdictiondict = {}
print('Perdiction Results: ')
for index,row in df_test.iterrows():
	lose = win = 0
	i = 0
	for feature in df_features:
		if i==0:
			lose = featuredict[feature][row[feature]]['Lose']
			win = featuredict[feature][row[feature]]['Win']
		else:
			try:
				lose *= featuredict[feature][row[feature]]['Lose']
			except:
				lose = 0
			try:
				win *= featuredict[feature][row[feature]]['Win']
			except:
				win = 0
		i += 1
	perdiction = 'Win' if (lose*pl/(l**3)) < (win*pw/(w**3)) else 'Lose'
	print("ID {}: {}".format(row['ID'],perdiction))
	perdictiondict[row['ID']] = perdiction 

# Accuracy, Precision, Recall, F1 score calculation
print('\nModel Evaluation: ')
TP = FP = TN = FN = 0
for index,row in df_test.iterrows():
	if perdictiondict[row['ID']] == 'Win' and row['Label'] == 'Win':
		TP += 1
	elif perdictiondict[row['ID']] == 'Win' and row['Label'] == 'Lose':
		FP += 1
	elif perdictiondict[row['ID']] == 'Lose' and row['Label'] == 'Lose':
		TN += 1
	elif perdictiondict[row['ID']] == 'Lose' and row['Label'] == 'Win':
		FN += 1
accuracy = (TP+TN)/(TP+FP+TN+FN)
precision = TP/(TP+FP)
recall = TP/(TP+FN)
F1 = 2*precision*recall / (precision+recall)
print("Accuracy: {}\nPrecision: {}\nRecall: {}\nF1 Score: {}".format(round(accuracy,3),round(precision,3),round(recall,3),round(F1,3)))


