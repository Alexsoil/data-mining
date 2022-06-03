import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import metrics
import word2vec

# Un-comment this line to set the seed for the creation of the random forest to a set value. Comment for randomized value each run
RANDOM_SWITCH = 100
# Set number of decision trees the algorithm will create. Default 100.
EST_NUM = 100
CRITERION = "gini"
MAX_DEPTH = None
# Number of CPU threads, for multithreaded tasks
THREADS = os.cpu_count()
# If 1, the code will run without loading the existing vectors.pkl and will not attempt to save data on it for later use.
NO_SAVE = 0

def data_info(df : pd.DataFrame):
    total = df.shape[0]
    # Count number of occurances for each label
    for i in range(1, 6):
        v_count = df['Score'].value_counts()[i]
        v_per = v_count / total
        print("Count of {num} Star Reviews: {count} ({percentage:.2%})".format(num=i, count=v_count, percentage=v_per))

def score_calculator(arr : np.array):
    t_prec = []
    t_rec = []
    # Calculate Precision/Recall for each class seperately
    for i in range(5):
        numerator = arr[i, i]
        t_prec.append(numerator / np.sum(arr[: , i]))
        t_rec.append(numerator / np.sum(arr[i, :]))
        print("{rating} Star - Precision: {prec:.2%} Recall: {rec:.2%}".format(rating=i+1, prec=t_prec[i], rec=t_rec[i]))
    # Macro averaging to get total Precision/Recall and calculate F1 Score
    f_prec = sum(t_prec)/len(t_prec)
    f_rec = sum(t_rec)/len(t_rec)
    f_score = 2 * ((f_prec * f_rec)/(f_prec + f_rec))
    # Print Results
    print("Total Precision: {prec:.2%}".format(prec=f_prec))
    print("Total Recall: {rec:.2%}".format(rec=f_rec))
    print("F1 Score: {score:.2%}".format(score=f_score))

try:
    print("Attempting to load Word2vec model...")
    word2vec.loadModel()
except:
    print("Loading failed, generating new model, please wait...")
    word2vec.CreateAndSaveModel()
print("Word2vec model loaded")

try:
    if (NO_SAVE != 0):
        raise Exception("No Save mode enabled")
    dataset = pd.read_pickle("dataset" + os.path.sep + "vectors.pkl")
    print("Pre-generated vectorized reviews loaded")
except:
    dataset = pd.read_csv("dataset" + os.path.sep + "amazon.csv")
    print("Vectorizing reviews...")
    for idx in tqdm(range(dataset.shape[0]), desc='Progress', ncols=100):
        dataset.at[idx, 'Text'] = word2vec.getVector(dataset['Text'].iloc[idx])
    print("Done")
    if (NO_SAVE == 0):
        print("Saving, please wait...")
        dataset.to_pickle("vectors.pkl")
        print("Vectorized reviews saved at dataset" + os.path.sep + "vectors.pkl")

# Print information about the contents
data_info(dataset)

# Initialize Random Forest, using 80% of the dataset for training and 20% for testing.
X = dataset['Text'].tolist()
y = dataset['Score'].tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=RANDOM_SWITCH)
forest = RandomForestClassifier(n_estimators=EST_NUM, criterion=CRITERION, max_depth=MAX_DEPTH, random_state=RANDOM_SWITCH, n_jobs=THREADS)
print("Random Forest Built")

# Fitting a model and making predictions
print("Fitting, please wait...")
forest.fit(X_train,y_train)
predictions = forest.predict(X_test)
print("Done")

# Creating a table to calculate recall and precision
y_pred = predictions.tolist()
results = np.zeros((5,5), dtype=np.int16)
try:
    for i in range(len(y_pred)):
        # Rows: Predicted (1 - 5) Columns: Actual (1 - 5)
        results[y_pred[i]-1, y_test[i]-1] += 1
except Exception as e:
    print(e)
    print("Something went wrong")

print("Results Table:")
print(results)

# Evaluating the model
acc = metrics.accuracy_score(y_test, predictions)
print("Accuracy: {accuracy:.2%}".format(accuracy=acc))

# Precision, Recall and F1 Score metrics
score_calculator(results)