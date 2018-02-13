import matplotlib.pyplot as plt
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
from sklearn.cluster import KMeans
import numpy as npx
import pickle
import os

def load_pickle(data_path):
    with open(data_path, 'rb') as handle:
        data_get = pickle.load(handle)
    return data_get

def save_picle(data_path, data):
    with open(data_path, 'wb') as handle:
        pickle.dump(data, handle)

def exists(path):
    """Test whether a path exists.  Returns False for broken symbolic links"""
    try:
        st = os.stat(path)
    except os.error:
        return False
    return True

def plot_data(data_path, index, save_path, filenames_get, embedding_size):
    path_model_load = data_path + "/" + filenames_get
    print "########"
    print path_model_load
    model = KeyedVectors.load(path_model_load)
    length = len(model.wv.vocab.keys())
    words = model.wv.vocab.keys()
    print filenames_get[1]
    X = npx.empty((length, 128))
    count = 0
    un_seen_node = 0
    for node in words:
        try:
            vec_one = model[node]
        except:
            vec_one = npx.random.rand(embedding_size)
            un_seen_node +=1

        #print vec_one.shape
        X[count, :] = vec_one
        count +=1
    y_pred = KMeans(n_clusters= 6, random_state= 2017, max_iter = 3000).fit_predict(X)
    X_embedded = None
    #kmeans.labels_
    #     print "do the TSNE"
    #     flag = exists(save_path)
    #     if not flag:
    #         X_embedded = TSNE(n_components=3).fit_transform(X)
    #         save_picle(save_path, X_embedded)
    #     else:
    #         X_embedded = load_pickle(save_path)
    return X, y_pred, X_embedded

data_path =  "/notebooks/logs_data/SDNE_embedding/"

filenames = ["node2vec_ppi_model__02_07_2018.txt", "node2vec_pos_model__02_07_2018.txt", "model__02_07_2018_blogs_2.txt"]
result_meta_nodes = []
embedding_size = 128
for file_path_1 in filenames:
    print "#######################"
    save_path = data_path + "data1"
    X, y_pred, X_embedded = plot_data(data_path, 0, save_path, file_path_1, embedding_size)
    from sklearn import metrics
    print file_path_1
    score_get =  metrics.calinski_harabaz_score(X, y_pred)
    result_meta_nodes.append([file_path_1[1],score_get])

# for the data visualization

# plt.plot( key_get, metanode2vec_value, 'go-', label='metanode2vec', linewidth=2)
# plt.plot( key_get, node2vec_value, 'ro-', label='node2vec', linewidth=2)
# plt.xticks(key_get)
#
# plt.xlabel("dimension d")
#
#
# plt.ylabel('Calinski Harabaz Score')
#
# # plt.suptitle('calinski harabaz score ')
# plt.legend()
#
# plt.savefig('calinski_harabaz_score_different_dimension_d', dpi = 100)
# plt.show()
