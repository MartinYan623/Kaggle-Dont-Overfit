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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
import numpy as np
import pylab as plot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit, StratifiedKFold
from sklearn.model_selection import GridSearchCV
# 读取训练集和测试集
train = pd.read_csv('data/train.csv')
test = pd.read_csv("data/test.csv")
train = train.drop('id', axis=1)

targets = train['target']
train = train.drop('target', axis=1)
id = test['id']
test = test.drop('id', axis=1)

x_train, x_validation, y_train, y_validation = train_test_split(train, targets, test_size=0.4, random_state=10)
clf = LogisticRegressionCV(class_weight='balanced', n_jobs=-1, penalty='l1', solver='liblinear')

# cross-validation
cv = ShuffleSplit(n_splits=5, test_size=.2, random_state=0)
skf = StratifiedKFold(n_splits=3, random_state=0, shuffle=False)
result = cross_val_score(clf, train, targets, cv=skf, scoring='f1')
print(result)


# param_grid = {"gamma":[0.001,0.01,0.1,1,10,100],
#              "C":[0.001,0.01,0.1,1,10,100]}
# print("Parameters:{}".format(param_grid))
#
# # 实例化一个GridSearchCV类
# grid_search = GridSearchCV(clf, param_grid, cv=5)
# grid_search.fit(x_train, y_train)
# print("Test set score:{:.2f}".format(grid_search.score(x_validation, y_validation)))
# print("Best parameters:{}".format(grid_search.best_params_))
# print("Best score on train set:{:.2f}".format(grid_search.best_score_))

def roc_auc(x_train, x_validation, y_train, y_validation):
    index = 0
    lw = 2
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    for lr in algorithms:
        y_score = lr.fit(x_train, y_train).predict_proba(x_validation)
        fpr, tpr, threshold = roc_curve(y_validation, y_score[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr,
                 lw=lw, label=str(index) + str('(area = %0.4f)' % roc_auc))
        index+=1
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

cw = 'balanced'
algorithms = [
    LogisticRegressionCV(random_state=1),
    RandomForestClassifier(min_samples_leaf=3, n_estimators=50, min_samples_split=10, max_depth=10),
    GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3),
    AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=8, class_weight=cw), n_estimators=50)
]

# full_predictions = []
# for alg in algorithms:
#     # Fit the algorithm using the full training data.
#     alg.fit(x_train, y_train)
#     # Predict using the test dataset.  We have to convert all the columns to floats to avoid an error.
#     predictions = alg.predict_proba(x_validation.astype(float))[:, 1]
#     full_predictions.append(predictions)
# predictions = (full_predictions[0] + full_predictions[1] + full_predictions[2] + + full_predictions[3]) / 4
# predictions[predictions > 0.5] = 1
# predictions[predictions <= 0.5] = 0
# predictions = predictions.astype(int)


lw = 2
plt.figure(figsize=(5, 5))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
y_score = clf.fit(x_train, y_train).predict_proba(x_validation)
fpr, tpr, threshold = roc_curve(y_validation, y_score[:, 1])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr,lw=lw, label=str('(area = %0.4f)' % roc_auc))
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

clf.fit(train, targets)
predictions = clf.predict(test)
submission = pd.DataFrame({
    "id": id,
    "target": predictions
})
print(submission)
submission.to_csv('/Users/martin_yan/Desktop/submission1.csv', index=False)
