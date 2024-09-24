#pip install 
#pip install matplotlib
#pip install pandas

import pandas as pd
import numpy as np
from sklearn import tree, metrics
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from scipy.io import arff


data,meta = arff.loadarff('./BaseThiago.arff')

attributes = meta.names()
data_value = np.asarray(data)


idade = np.asarray(data['idade']).reshape(-1,1)
saldo = np.asarray(data['saldo']).reshape(-1,1)
features = np.concatenate((saldo , idade ),axis=1)
target = data['situacao']


Arvore = DecisionTreeClassifier(criterion='entropy').fit(features, target)

plt.figure(figsize=(10, 6.5))
tree.plot_tree(Arvore,feature_names=['idade','saldo'],class_names=['Devendo', 'Normal'],
                   filled=True, rounded=True)
plt.show()

fig, ax = plt.subplots(figsize=(25, 10))
metrics.ConfusionMatrixDisplay.from_estimator(Arvore,features,target,display_labels=['Devendo', 'Normal'], values_format='d', ax=ax)
plt.show()