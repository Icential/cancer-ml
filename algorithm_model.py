
# predefined classification algorithm-based model using sklearn
# ~88% accuracy on average


# imports here
import pandas as pd
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


"""Setting up datasets"""
# load csv dataset
df = pd.read_csv("cancer.csv")

# partition dataset into training and validation sets
x = pd.get_dummies(df.drop(['LUNG_CANCER', 'SMOKING'], axis=1))
y = df['LUNG_CANCER']
x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.2, random_state=1)


"""Evaluate given models (by mean accuracy and standard deviation)"""
# algorithm models (pick highest mean and lowest standard deviation)
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVC', SVC(gamma='auto')))

# evaluate models
results, names = [], []
print('(Name, Mean, Standard Deviation)')
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


# compare models using plt
pyplot.boxplot(results, labels=names)
pyplot.title('Model Comparisons')
pyplot.show()


"""Choose and further evaluate model"""
# make predictions on validation dataset (Choose SVC)
model = LogisticRegression(solver='liblinear', multi_class='ovr')
model.fit(x_train, y_train)
predictions = model.predict(x_validation)


# evaluate predictions
print('\nOverall Accuracy: ', accuracy_score(y_validation, predictions))
print('\nConfusion Matrix:\n', confusion_matrix(y_validation, predictions))
print('\nClassification Table:\n', classification_report(y_validation, predictions)) # f1-score proportional to precision and recall