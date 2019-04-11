import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression  # 线性回归
from sklearn.cross_validation import KFold  # 交叉验证库，将测试集进行切分交叉验证取平均
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import numpy as np
import pylab as plot

# 读取训练集和测试集
train = pd.read_csv('data/train.csv')
test = pd.read_csv("data/test.csv")
train = train.drop('id', axis=1)

targets = train['target']
train = train.drop('target', axis=1)
id = test['id']
test = test.drop('id', axis=1)

rf = RandomForestClassifier(min_samples_leaf=3, n_estimators=50, min_samples_split=10, max_depth=10)

cw = 'balanced'
algorithms = [
    LogisticRegressionCV(random_state=1),
    RandomForestClassifier(min_samples_leaf=3, n_estimators=50, min_samples_split=10, max_depth=10),
    GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3),
    AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=8, class_weight=cw), n_estimators=50)
]

full_predictions = []
for alg in algorithms:
    # Fit the algorithm using the full training data.
    alg.fit(train, targets)
    # Predict using the test dataset.  We have to convert all the columns to floats to avoid an error.
    predictions = alg.predict_proba(test.astype(float))[:, 1]
    full_predictions.append(predictions)
predictions = (full_predictions[0] + full_predictions[1] + full_predictions[2] + + full_predictions[3]) / 4
predictions[predictions > 0.5] = 1
predictions[predictions <= 0.5] = 0
predictions = predictions.astype(int)

submission = pd.DataFrame({
    "id": id,
    "target": predictions
})
print(submission)
submission.to_csv('/Users/martin_yan/Desktop/submission1.csv', index=False)
