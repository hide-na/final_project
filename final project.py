import numpy as np
import numpy.random as random
import scipy as sp
from pandas import Series, DataFrame
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
# %matplotlib inline

import sklearn

# %precision 3

import requests, zipfile
import io


# Import data
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data'
res = requests.get(url).content

# Data into an object of DataFrame
mushroom = pd.read_csv(io.StringIO(res.decode('utf-8')), header=None)

# Set label on data columns
mushroom.columns = ['classes','cap_shape', 'cap_surface','cap_color', 'odor','bruises',
                    'gill_attachment','gill_spacing','gill_size', 'gill_color', 'stalk_shape', 'stalk_root',
                    'stalk_surface_above_ring', 'stalk_surface_below_ring', 'stalk_color_above_ring',
                    'stalk_color_below_ring', 'veil_type', 'veil_color', 'ring_number', 'ring_type', 'spore_print_color',
                    'population','habitat']
# Show first 5 rows
print(mushroom.head())
# Check number of rows, columns and missing data
print('Data type:{}'.format(mushroom.shape))
print('Number of missing data: {}'.format(mushroom.isnull().sum().sum()))

# Choose 4 explanatory variables (gill_color, gill_attachment, odor, cap_color)
mushroom_dummy = pd.get_dummies(mushroom[['gill_color','gill_attachment','odor','cap_color']])
print(mushroom_dummy.head())

# Objective variable (0 or 1)
mushroom_dummy['fig'] = mushroom['classes'].map(lambda x:1 if x == 'p' else 0)

# Entropy
print(mushroom_dummy.groupby(['cap_color_c','fig'])['fig'].count().unstack)
print(mushroom_dummy.groupby(['gill_color_b','fig'])['fig'].count().unstack)


print(-(0.5 * np.log2(0.5) + 0.5 * np.log2(0.5)))
print(-(0.001 * np.log2(0.001) + 0.999 * np.log2(0.999)))

def calc_entropy(p):
    return - (p *np.log(p) + (1 -p) * np.log2(1-p))

# Change the valud of p by 0.01 from 0.001 to 0.999
p = np.arange(0.001, 0.999, 0.01)

# Draw a graph
plt.plot(p, calc_entropy(p))
plt.xlabel('prob')
plt.ylabel('entropy')
plt.grid(True)

mushroom_dummy.groupby('fig')['fig'].count()

entropy_init = -(0.518 * np.log2(0.518) + 0.482 * np.log2(0.482))
print('the valud of initial entropy of poisonous mushroom: {:.3f}'.format(entropy_init))

# Information gain
mushroom_dummy.groupby(['cap_color_c', 'fig'])['fig'].count().unstack()

# Entropy when cap_color is not c
p1 = 4176 / (4176 + 3904)
p2 = 1 - p1
entropy_c0 = -(p1*np.log2(p1) + p2*np.log2(p2))
print('entropy_c0: {:.3f}'.format(entropy_c0))

# Entropy when cap_color is c
p1 = 32 / (32 + 12)
p2 = 1 - p1
entropy_c1 = -(p1*np.log2(p1) + p2*np.log2(p2))
print('entropy_c1: {:.3f}'.format(entropy_c1))


entropy_after = (4176 + 3904)/ 8124 * entropy_c0 + (32 + 12)/ 8124 * entropy_c1
print('Average entropy after dividing data: {:.3f}'.format(entropy_after))

print('information gain from dviding cap_color: {:.3f}'.format(entropy_init - entropy_after))

mushroom_dummy.groupby(['gill_color_b', 'fig'])['fig'].count().unstack()

# Entropy when gill_color is not b
p1 = 4208 / (4208 + 2188)
p2 = 1 - p1
entropy_b0 = -(p1*np.log2(p1) + p2*np.log2(p2))

# Entropy when gill_color is b
p1 = 0 / (0 + 1728)
p2 = 1 - p1
entropy_b1 = -(p2*np.log2(p2))
entropy_after = (4208 + 2188)/ 8124 * entropy_b0 + (0 + 1728)/ 8124 * entropy_b1
print('Information gain after dividing gill_color: {:.3f}'.format(entropy_init - entropy_after))


# Decision tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Devide data
X = mushroom_dummy.drop('fig', axis=1)
y = mushroom_dummy['fig']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Initializatoin of decision tree class and learning
model = DecisionTreeClassifier(criterion='entropy', max_depth=13, random_state=0)
model.fit(X_train, y_train)

print('Accuracy rate(train):{:.3f}'.format(model.score(X_train, y_train)))
print('Accuracy rate(test):{:.3f}'.format(model.score(X_test, y_test)))

# Show graph of decision tree
from sklearn import tree
import pydotplus

from sklearn.externals.six import StringIO
from IPython.display import Image

dot_data = StringIO()
tree.export_graphviz(model, out_file=dot_data)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
