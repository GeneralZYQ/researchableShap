#This is used for testing import

import shap
import pickle 
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.image as mpimg


loans = pd.read_csv('loan_data.csv')
cat_feats = ['purpose']
final_data = pd.get_dummies(loans,columns=cat_feats,drop_first=True)

X = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

X_train_demo = X_train.iloc[range(0, 100), :]
y_train_demo = y_train.iloc[range(0,100)]

X_test_demo = X_test.iloc[range(0,10), :]
y_test_demo = y_test.iloc[range(0,10)]

loaded_model = pickle.load(open('tree_model.sav', 'rb'))
loaded_model.fit(X_train_demo,y_train_demo)


explainer = shap.TreeExplainer(loaded_model)
shap_values = explainer.shap_values(X_train_demo)

# result = loaded_model.score(X_test, Y_test)




# print(plots)


print('test')