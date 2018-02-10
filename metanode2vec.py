import numpy as np
import networkx as nx
import random
from sklearn.metrics import jaccard_similarity_score
from binning_algorithm import binning
from gensim.models import Word2Vec
from evaludation_model import mode_evaludatoin
import pickle
import os
import json
from gensim.models.keyedvectors import KeyedVectors
import scipy.io as sio
import numpy as np
import networkx as nx
from scipy.sparse import csc_matrix
import pandas as pd

embedding_size = 128
class Graph_type():
    def __init__(self, nx_G, is_directed, p, q, featur_vector_similarity,  binning_information, bining_second_order):
        self.G = nx_G
        self.is_directed = is_directed
        self.p = p
        self.q = q
        self.binning = binning_information
        self.order_order_infor = bining_second_order
        self.featur_vector_similarity = featur_vector_similarity
        self.similarity_nodes = 0.0
        #self.degree_bin = degree_bin

    def node2vec_walk(self, walk_length, start_node):
        '''
        Simulate a random walk starting from start node.
        '''
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges
        binning_infor = self.binning

        walk = [start_node]
        #root_degree = G.degree(start_node)
        root_bin_type = binning_infor[start_node]
        curr_type = root_bin_type
        count = 0
        sample_count = 0
        bining_second_order = self.order_order_infor
        featur_vector_similarity = self.featur_vector_similarity
        re_sample = False

        binning_infor = self.binning

        while len(walk) < walk_length:
            cur = walk[-1]
            root_bin_infor = binning_infor[cur]
            cur_nbrs = sorted(G.neighbors(cur))
            if re_sample:
                sample_count +=1 # in case there is one types, to dead loop.
            else:
                sample_count = 0
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    next_node = cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])]
                else:
                    prev = walk[-2]
                    next_node = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0],
                                                    alias_edges[(prev, cur)][1])]
                node_bin_infor = binning_infor[next_node]

                common_elements = len([item for item in root_bin_infor if item not in node_bin_infor])#len(root_bin_infor - (root_bin_infor - node_bin_infor))
                combine_elements = len(set( root_bin_infor + node_bin_infor ))#float(len(root_bin_infor | node_bin_infor))
                #next_degree = G.degree(next_node)
                next_type = binning_infor[next_node]
                key_second_order = (cur, next_node)
                key_second_oder_2 = (next_node, cur)

                if key_second_order in bining_second_order:
                    second_order_similarity = bining_second_order[key_second_order]
                elif key_second_oder_2 in bining_second_order:
                    second_order_similarity = bining_second_order[key_second_oder_2]
                else:
                    second_order_similarity = 0.0

                if key_second_order in featur_vector_similarity:
                    node2vec_similarity = featur_vector_similarity[key_second_order]
                elif key_second_oder_2 in featur_vector_similarity:
                    node2vec_similarity = featur_vector_similarity[key_second_oder_2]
                else:
                    node2vec_similarity = 0.0

                similarity_nodes = 1.0000 - (float(common_elements)/combine_elements + second_order_similarity + node2vec_similarity)/3.0

                if similarity_nodes>0.65:
                    count += 1
                else:
                    count = 0
                    curr_type = next_type # update the current type

                if count <= 5 or sample_count>3:
                    walk.append(next_node)
                    re_sample = False
                else:
                    re_sample = True
            else:
                break
        return walk

    def simulate_walks(self, num_walks, walk_length):
        '''
        Repeatedly simulate random walks from each node.
        '''
        G = self.G
        walks = []
        nodes = list(G.nodes())
        print 'Walk iteration:'
        for walk_iter in range(num_walks):
            print str(walk_iter+1), '/', str(num_walks)
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))
        return walks


    def get_alias_edge(self, src, dst):
        '''
        Get the alias edge setup lists for a given edge.
        '''
        G = self.G
        p = self.p
        q = self.q
        binning_infor = self.binning

        unnormalized_probs = []
        root_bin_infor = binning_infor[src]
        #root_bin_infor = binning_infor[root_degree]
        bining_second_order = self.order_order_infor
        featur_vector_similarity = self.featur_vector_similarity

        for dst_nbr in sorted(G.neighbors(dst)):
            node_bin_infor = binning_infor[dst_nbr]
            common_elements = len([item for item in root_bin_infor if item not in node_bin_infor])#len(root_bin_infor - (root_bin_infor - node_bin_infor))
            combine_elements = len(set( root_bin_infor + node_bin_infor ))#float(len(root_bin_infor | node_bin_infor))
            # get the second order information:
            key_second_order = (dst, dst_nbr)
            key_second_oder_2 = (dst_nbr, dst)
            if key_second_order in bining_second_order:
                second_order_similarity = bining_second_order[key_second_order]
            elif key_second_oder_2 in bining_second_order:
                second_order_similarity = bining_second_order[key_second_oder_2]
            else:
                second_order_similarity = 0.0

            # for node2vec feature vector similarity
            if key_second_order in featur_vector_similarity:
                node2vec_similarity = featur_vector_similarity[key_second_order]
            elif key_second_oder_2 in featur_vector_similarity:
                node2vec_similarity = featur_vector_similarity[key_second_oder_2]
            else:
                 node2vec_similarity = 0.0

            self.similarity_nodes = 1.0000 - (float(common_elements)/combine_elements + second_order_similarity + node2vec_similarity)/3.0

            prob_tranform = 1.0000 - self.similarity_nodes # 1.0001 - jaccard_similarity_score(root_bin_infor, node_bin_infor)

            if dst_nbr == src:
                unnormalized_probs.append(prob_tranform)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(prob_tranform)
            else:
                unnormalized_probs.append(prob_tranform)

        norm_const = sum(unnormalized_probs)
        normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        '''

        G = self.G
        is_directed = self.is_directed

        alias_nodes = {}
        count = 0
        for node in G.nodes():
            #unnormalized_probs = [G[node][nbr][0]['weight'] for nbr in sorted(G.neighbors(node))]
            unnormalized_probs = [1 for nbr in sorted(G.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}
        triads = {}
        print "do the data processing"
        if is_directed:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                count +=1
                if count%100==0:
                    print "the processing number is %f"
                    print float(int(count))/len(G.edges())

        else:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])
                count +=1
                if count%100==0:
                    print "the processing number is %f ----"
                    print float(int(count))/len(G.edges())

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return

def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q

def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    K = len(J)
    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]


def nodes_connected(u, v):
    return u in G.neighbors[v]


def learn_embeddings(walks, path_1):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    flag = exists(path_1)
    if flag:
        model = KeyedVectors.load(path_1)
        print "################"
        print "load the model directly"
    else:
        walks = [map(str, walk) for walk in walks]
        model = Word2Vec(walks, size= embedding_size, window= 5, min_count=5, sg=1, workers= 8, iter= 300)
        model.save(path_1)
    return model

def get_auc(G, graph_type, labels, featur_vector_similarity, binning_nodes, second_order_get = None):
    pathModel = os.getcwd()  #
    if graph_type== "nodeTag2Vec":
        G_graph_tag = Graph_type(G, False, 1, 1, featur_vector_similarity, binning_nodes, second_order_get)
        path_model = os.path.join(pathModel, "methnodes_model_02_10_2018_blog_data.txt")
    sequence_data_tag = None
    if not exists(path_model):
        G_graph_tag.preprocess_transition_probs()
        sequence_data_tag = G_graph_tag.simulate_walks(10, 80)

    model = learn_embeddings(sequence_data_tag, path_model)
    print "do the evaludation"
    obj_model = mode_evaludatoin(model, G, embedding_size)
    scores_get = obj_model.mutli_lables(labels, embedding_size)
    return scores_get

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

if __name__ == '__main__':
    print "load graph begin"
    path_file = os.getcwd()
    print path_file
    # data_path = path_file + '/data/Homo_sapiens.mat'  #  methnodes_model_02_10_2018_update_node2vec.txt
    # path_2vec_model = "/Users/weizhili/Desktop/data_docker/logs_data/SDNE_embedding/node2vec_ppi_model__02_07_2018.txt"

    blog_data = True
    if not blog_data:
        #data_path = path_file + '/data/Homo_sapiens.mat'  #  methnodes_ppi_02_10_2018_update_node2vec.txt
        #path_2vec_model = "/Users/weizhili/Desktop/data_docker/logs_data/SDNE_embedding/node2vec_ppi_model__02_07_2018.txt"

        data_path = path_file + '/data/POS.mat' # methnodes_pos_02_10_2018_update_node2vec.txt
        path_2vec_model = "/Users/weizhili/Desktop/data_docker/logs_data/SDNE_embedding/node2vec_pos_model__02_07_2018.txt"  # methnodes_model_02_09_2018_pos_data.txt
        mat_contents = sio.loadmat(data_path)
        labels = mat_contents["group"].todense()
        print labels
        network = mat_contents["network"].todense()

        G=nx.from_numpy_matrix(network, create_using=nx.MultiDiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]][0]['weight'] = 1

    if blog_data:
        data_path = path_file + "/data/BlogCatalog-dataset/data/"

        os.chdir(data_path)
        filenames = [x for x in os.listdir(data_path) if x.endswith('.csv') and os.path.getsize(x) > 0]
        labels = pd.read_csv(data_path + filenames[1], names = ["nodes", "label"]) #df.replace({"col1": di})
        G = nx.read_edgelist(data_path + filenames[0], delimiter=",", data=[("weight", int)])
        path_2vec_model = "/Users/weizhili/Desktop/data_docker/logs_data/SDNE_embedding/model__02_07_2018_blogs_2.txt"  # methnodes_model_02_10_2018_blog_data.txt

    print data_path

    # read the multiplables
    binning_obj = binning(G, path_2vec_model)
    binning_nodes = binning_obj.binning_graph()
    bining_second_order_infor = binning_obj.second_order_get()

    if not blog_data:
        featur_vector_similarity = binning_obj.feature_combine()
    if blog_data:
        print "similarithy"
        featur_vector_similarity = binning_obj.feature_combine_blog()

    # do the embedding
    print "model evaludation"
    auc_value, cross_vaidation_values = get_auc(G, "nodeTag2Vec", labels, featur_vector_similarity, binning_nodes, bining_second_order_infor)
    print path_2vec_model
