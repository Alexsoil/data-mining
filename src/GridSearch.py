import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

THREADS = os.cpu_count()

parameters = {
    'n_estimators': [100],
    'max_depth': [None],
    'criterion': ['gini', 'entropy', 'log_loss']
}

try:
    dataset = pd.read_pickle("dataset" + os.path.sep + "vectors.pkl")
    print("Pre-generated vectorized reviews loaded")
except:
    print("Dataset failed to load or doesn't exist")
    exit(1)

X = dataset['Text'].tolist()
y = dataset['Score'].tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=100)

cv = GridSearchCV(RandomForestClassifier(), parameters, cv=5, verbose=1, n_jobs=THREADS)

cv.fit(X_train, y_train)

print(f'Best parameters are: {cv.best_params_}')
print("\n")
mean_score = cv.cv_results_['mean_test_score']
std_score = cv.cv_results_['std_test_score']
params = cv.cv_results_['params']
for mean,std,params in zip(mean_score,std_score,params):
    print(f'{round(mean,3)} + or -{round(std,3)} for the {params}')