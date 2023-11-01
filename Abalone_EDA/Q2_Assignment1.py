# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 15:27:52 2023

@author: 15485
"""

import pandas as pd
import matplotlib.pyplot as plt
import warnings 

warnings.filterwarnings('ignore')

#%%
df = pd.read_csv(r"C:\Users\15485\Desktop\UWaterloo_Academics\ECE657A\Assignments\Assignment1\abalone.csv", names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 
                      'Sucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings'], sep = ',')

#%%
# Check any relationship between Sex and Rings
plt.scatter(y=df['Sex'], x=df['Rings'])
plt.xlabel('Rings')
plt.ylabel('Sex')
plt.show()

# This feature is not a very helpful tool to predict the Rings and hence we can drop it. 

#%%

X = df.iloc[:, 1:8]  # Removing sex feature  

y = df.iloc[:, 8]   #using Rings as a output for supervised learning
list_scores = []

# Split the dataset into 80:20
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Apply KNN classification
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
knn_initial_score = knn.score(X_test, y_test)
print(knn_initial_score)

#%%
# Balance the training dataset
from imblearn.over_sampling import RandomOverSampler
os = RandomOverSampler(random_state=1)
X_train_sampled, y_train_sampled = os.fit_resample(X_train, y_train)

#%%
# Apply Z-score 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_sampled.iloc[:, 1:] = sc.fit_transform(X_train_sampled.iloc[:, 1:])


#%%
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

def model_training(K):
    model_score = []
    accuracy_list = []
    # Split the 80% training dataset using KFold
    kf = StratifiedKFold(n_splits=5)

    for train_index, test_index in kf.split(X_train_sampled, y_train_sampled):
        X_train, X_test = X_train_sampled.iloc[train_index,:], X_train_sampled.iloc[test_index,:] 
        y_train, y_test = y_train_sampled[train_index] , y_train_sampled[test_index]

        # create model for every fold
        knn = KNeighborsClassifier(n_neighbors=K)
        knn.fit(X_train, y_train)
        model_score.append(knn.score(X_test, y_test))
        pred_values = knn.predict(X_test)

        acc = accuracy_score(pred_values , y_test)
        accuracy_list.append(acc)

    avg_accuracy = sum(accuracy_list)/K
    return avg_accuracy

#%%
for i in range(5, 15):
    acc_list = []
    avg_acc = model_training(i)
    print(i, avg_acc)
    plt.scatter(x=i, y=avg_acc*100)
    plt.xlabel("Values of K across all folds")
    plt.ylabel("Mean Validation Accuracy")
    
#%%
# Running KNN using K=10
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
knn_initial_score = knn.score(X_test, y_test)

print(knn_initial_score)

#%%
#KNN improvement using Weighted KNN
import numpy as np
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# knn = KNeighborsClassifier(n_neighbors=10)

knn = KNeighborsClassifier(metric='wminkowski', p=1, n_neighbors=10, weights='distance' ,metric_params={'w': np.random.random(X_train.shape[1])})
knn.fit(X_train, y_train)
knn_initial_score = knn.score(X_test, y_test)
print(knn_initial_score)

