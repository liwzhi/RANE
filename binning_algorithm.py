import pandas as pd
import networkx as nx

def DistJaccard(list_nodes_1, list_nodes_2):
    return float(len(list_nodes_1 & list_nodes_2)) / len(list_nodes_1 | list_nodes_2)

import math
def cosine_similarity(v1,v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)

class binning():
    def __init__(self, G, path_model_load =None):
        self.G = G
        self.dict_lookup = None
        self.second_order = None
        self.path_model_load = path_model_load

    def second_order_get(self):
        print "do the second binning"
        G = self.G
        neighbors_common = {}
        for node in G.nodes():
            neighbors_nodes = list(G.neighbors(node))
            for neighbor in neighbors_nodes:
                neighbor_seond_order = list(G.neighbors(neighbor))

                common_elements = len([item for item in neighbors_nodes if item not in neighbor_seond_order])#len(root_bin_infor - (root_bin_infor - node_bin_infor))                print list(neighbor_seond_order)
                combine_elements = len(set( neighbors_nodes + neighbor_seond_order ))#float(len(root_bin_infor | node_bin_infor))
                prob_tranform = float(common_elements)/combine_elements #1.0000 - jaccard_similarity_score
                key1 = (node, neighbor)
                key2 = (neighbor, node)
                if key1 not in neighbors_common and key2 not in neighbors_common:
                    neighbors_common[key1] = prob_tranform

        self.second_order = neighbors_common
        return neighbors_common

    def binning_graph(self):
        # do the first order
        G = self.G
        degree_list = []
        node_list = []
        sum_degree = 0
        for node in G.nodes():
            degree = G.degree(node)
            sum_degree += degree
            degree_list.append(degree)
            node_list.append(node)
        pd_degree = pd.DataFrame()

        pd_degree["degree"] = degree_list
        bin_number = 4#sum_degree/len(G.nodes()) + 1 # get the averages nodes number
        print "the average degree is %d" % bin_number

        pd_degree['rank'] = pd_degree['degree'].rank(method='first')
        out = pd.qcut(pd_degree["rank"], bin_number, labels = range(bin_number))
        bin_data = out.values
        dict_lookup = {}
        high_degree_nodes = []
        for index in range(len(bin_data)):
            bin_result = bin_data[index]
            node_item = node_list[index]
            dict_lookup[node_item] = [bin_result] # CAN BE DIFFERENT TYPES, degree to bin results
        self.dict_lookup = dict_lookup
        return dict_lookup

    def feature_combine(self):
        from gensim.models.keyedvectors import KeyedVectors
        path_model_load = self.path_model_load
        featur_vector_similarity = {}

        if path_model_load:
            G = self.G
            node2vec_model = KeyedVectors.load(path_model_load)
            for node in G.nodes():
                neighbors_nodes = list(G.neighbors(node))
                root_node_vector = node2vec_model[str(node)]
                for neighbor in neighbors_nodes:
                    neighbor_vector = node2vec_model[str(neighbor)]
                    similarity = cosine_similarity(root_node_vector, neighbor_vector)
                    key1 = (node, neighbor)
                    key2 = (neighbor, node)
                    if key1 not in featur_vector_similarity and key2 not in featur_vector_similarity:
                        featur_vector_similarity[key1] = similarity
        return featur_vector_similarity

    def feature_combine_blog(self):
        from gensim.models.keyedvectors import KeyedVectors
        path_model_load = self.path_model_load
        featur_vector_similarity = {}
        G = self.G
        # create the mapping maps
        nodes_map = {}
        count = 0
        for item in list(G.nodes()):
            if item not in nodes_map:
                nodes_map[int(item)] = count
                count += 1
        # get the nodes_map, 1 --> count (index)

        G =nx.relabel_nodes(G, nodes_map) # change the index, from hash id to index.

        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

        if path_model_load:
            print "begin to load the model"
            node2vec_model = KeyedVectors.load(path_model_load)
            for node in G.nodes():
                neighbors_nodes = list(G.neighbors(node))
                root_node_vector = node2vec_model[str(node)]
                for neighbor in neighbors_nodes:
                    neighbor_vector = node2vec_model[str(neighbor)]
                    similarity = cosine_similarity(root_node_vector, neighbor_vector)
                    key1 = (node, neighbor)
                    key2 = (neighbor, node)
                    if key1 not in featur_vector_similarity and key2 not in featur_vector_similarity:
                        featur_vector_similarity[key1] = similarity
            print "finish the loading the model"
        return featur_vector_similarity
