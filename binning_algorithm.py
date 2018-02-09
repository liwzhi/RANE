import pandas as pd

def DistJaccard(list_nodes_1, list_nodes_2):
    return float(len(list_nodes_1 & list_nodes_2)) / len(list_nodes_1 | list_nodes_2)

class binning():
    def __init__(self, G):
        self.G = G
        self.dict_lookup = None
        self.second_order = None

    def second_order_get(self):
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
        bin_number = sum_degree/len(G.nodes()) + 1 # get the averages nodes number
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
