
import pandas as pd
import os
import networkx as nx
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, cross_validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score


model_path = "/notebooks/logs_data/SDNE_embedding/metahnodes2vec_02_08_blog_data.txt"
data_path = "/notebooks/logs_data/SDNE_embedding/BlogCatalog-dataset/data/"
os.chdir(data_path)
filenames = [x for x in os.listdir(data_path) if x.endswith('.csv') and os.path.getsize(x) > 0]
labels = pd.read_csv(data_path + filenames[1], names = ["nodes", "label"]) #df.replace({"col1": di})
G = nx.read_edgelist(data_path + filenames[0], delimiter=",", data=[("weight", int)])
nodes_map = {}
count = 0
for item in list(G.nodes()):
    if item not in nodes_map:
        nodes_map[int(item)] = count
        count += 1

G =nx.relabel_nodes(G, nodes_map)

for edge in G.edges():
    G[edge[0]][edge[1]]['weight'] = 1

model = KeyedVectors.load(model_path)
nodes_index = list(labels.nodes)
nodes_mapping = []
for item in nodes_index:
    nodes_mapping.append(nodes_map[item])

embedding_size = 128
X = np.empty((len(nodes_mapping), embedding_size))
count = 0
un_seen_node = 0



for node in nodes_mapping:
    try:
        vec_one = model[str(node)]
    except:
        vec_one = np.random.rand(embedding_size)
        un_seen_node +=1

    X[count, :] = vec_one
    count +=1

lb = preprocessing.LabelBinarizer()
labels_get = list(labels.label)
labels_get =[[x] for x in labels_get]
y = MultiLabelBinarizer().fit_transform(labels_get)
list_mico = []
list_maco = []
items = [p/10.0 for p in range(1, 10)]

for item in items:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= item, random_state=51)
    clf = LogisticRegression() #C= 1, penalty = "l2", tol=0.01)
    y_score = OneVsRestClassifier(clf).fit(X_train, y_train).predict(X_test)

    item_preict = []
    for item in y_score:
        if item.any():
            item_preict.append(item)
    all_zeros = not np.any(y_score)
    micro_f1 = f1_score(y_test, y_score, average='micro')
    macro_f2 = f1_score(y_test, y_score, average='macro')
    print micro_f1
    print macro_f2
    list_mico.append(micro_f1)
    list_maco.append(macro_f2)
