import networkx as nx
import time
import pickle
import os

import numpy as np
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, cross_validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
import time
timestr = time.strftime("%Y%m%d-%H%M%S")



def exists(path):
    """Test whether a path exists.  Returns False for broken symbolic links"""
    try:
        st = os.stat(path)
    except os.error:
        return False
    return True

def load_pickle(data_path):
    with open(data_path, 'rb') as handle:
        data_get = pickle.load(handle)
    return data_get

def save_picle(data_path, data):
    with open(data_path, 'wb') as handle:
        pickle.dump(data, handle)


class DBLPDataLoader:
    def __init__(self, graph_file):
        self.g = graph_file
        self.num_of_nodes = self.g.number_of_nodes()
        self.num_of_edges = self.g.number_of_edges()
        self.edges_raw = self.g.edges(data=True)
        self.nodes_raw = self.g.nodes(data=True)

        self.edge_distribution = np.array([attr['weight'] for _, _, attr in self.edges_raw], dtype=np.float32)
        self.edge_distribution /= np.sum(self.edge_distribution)
        self.edge_sampling = AliasSampling(prob=self.edge_distribution)
        self.node_negative_distribution = np.power(
            np.array([self.g.degree(node, weight='weight') for node, _ in self.nodes_raw], dtype=np.float32), 0.75)
        self.node_negative_distribution /= np.sum(self.node_negative_distribution)
        self.node_sampling = AliasSampling(prob=self.node_negative_distribution)

        self.node_index = {}
        self.node_index_reversed = {}
        for index, (node, _) in enumerate(self.nodes_raw):
            self.node_index[node] = index
            self.node_index_reversed[index] = node
        self.edges = [(self.node_index[u], self.node_index[v]) for u, v, _ in self.edges_raw]

    def fetch_batch(self, batch_size=16, K=10, edge_sampling='atlas', node_sampling='atlas'):
        if edge_sampling == 'numpy':
            edge_batch_index = np.random.choice(self.num_of_edges, size=batch_size, p=self.edge_distribution)
        elif edge_sampling == 'atlas':
            edge_batch_index = self.edge_sampling.sampling(batch_size)
        elif edge_sampling == 'uniform':
            edge_batch_index = np.random.randint(0, self.num_of_edges, size=batch_size)
        u_i = []
        u_j = []
        label = []
        for edge_index in edge_batch_index:
            edge = self.edges[edge_index]
            if self.g.__class__ == nx.Graph:
                if np.random.rand() > 0.5:      # important: second-order proximity is for directed edge
                    edge = (edge[1], edge[0])
            u_i.append(edge[0])
            u_j.append(edge[1])
            label.append(1)
            for i in range(K):
                while True:
                    if node_sampling == 'numpy':
                        negative_node = np.random.choice(self.num_of_nodes, p=self.node_negative_distribution)
                    elif node_sampling == 'atlas':
                        negative_node = self.node_sampling.sampling()
                    elif node_sampling == 'uniform':
                        negative_node = np.random.randint(0, self.num_of_nodes)
                    if not self.g.has_edge(self.node_index_reversed[negative_node], self.node_index_reversed[edge[1]]):
                        break
                u_i.append(edge[0])
                u_j.append(negative_node)
                label.append(-1)
        return u_i, u_j, label

    def embedding_mapping(self, embedding):
        return {node: embedding[self.node_index[node]] for node, _ in self.nodes_raw}


class AliasSampling:

    # Reference: https://en.wikipedia.org/wiki/Alias_method

    def __init__(self, prob):
        self.n = len(prob)
        self.U = np.array(prob) * self.n
        self.K = [i for i in range(len(prob))]
        overfull, underfull = [], []
        for i, U_i in enumerate(self.U):
            if U_i > 1:
                overfull.append(i)
            elif U_i < 1:
                underfull.append(i)
        while len(overfull) and len(underfull):
            i, j = overfull.pop(), underfull.pop()
            self.K[j] = i
            self.U[i] = self.U[i] - (1 - self.U[j])
            if self.U[i] > 1:
                overfull.append(i)
            elif self.U[i] < 1:
                underfull.append(i)

    def sampling(self, n=1):
        x = np.random.rand(n)
        i = np.floor(self.n * x)
        y = self.n * x - i
        i = i.astype(np.int32)
        res = [i[k] if y[k] < self.U[i[k]] else self.K[i[k]] for k in range(n)]
        if n == 1:
            return res[0]
        else:
            return res


import tensorflow as tf

class LINEModel:
    def __init__(self, args):
        self.u_i = tf.placeholder(name='u_i', dtype=tf.int32, shape=[args.batch_size * (args.K + 1)])
        self.u_j = tf.placeholder(name='u_j', dtype=tf.int32, shape=[args.batch_size * (args.K + 1)])
        self.label = tf.placeholder(name='label', dtype=tf.float32, shape=[args.batch_size * (args.K + 1)])
        self.embedding = tf.get_variable('target_embedding', [args.num_of_nodes, args.embedding_dim],
                                         initializer=tf.random_uniform_initializer(minval=-1., maxval=1.))
        self.u_i_embedding = tf.matmul(tf.one_hot(self.u_i, depth=args.num_of_nodes), self.embedding)
        if args.proximity == 'first-order':
            self.u_j_embedding = tf.matmul(tf.one_hot(self.u_j, depth=args.num_of_nodes), self.embedding)
        elif args.proximity == 'second-order':
            self.context_embedding = tf.get_variable('context_embedding', [args.num_of_nodes, args.embedding_dim],
                                                     initializer=tf.random_uniform_initializer(minval=-1., maxval=1.))
            self.u_j_embedding = tf.matmul(tf.one_hot(self.u_j, depth=args.num_of_nodes), self.context_embedding)

        self.inner_product = tf.reduce_sum(self.u_i_embedding * self.u_j_embedding, axis=1)
        self.loss = -tf.reduce_mean(tf.log_sigmoid(self.label * self.inner_product))
        self.learning_rate = tf.placeholder(name='learning_rate', dtype=tf.float32)
        # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)


class data_obj:
    def __init__(self, proximity= 'second-order', embedding_dim = 128):
        self.embedding_dim = embedding_dim
        self.batch_size = 128
        self.K = 5
        self.proximity = proximity
        self.learning_rate = 0.025

        self.mode = 'train'
        self.num_batches = 1000
        self.total_graph = True

class run_model():
    def get_value(self, nodes, normalized_embedding, embedding_size = 128):
        X_averge = np.empty([len(nodes), embedding_size])
        X_hadamard = np.empty([len(nodes), embedding_size])
        X_weighted_L1 = np.empty([len(nodes), embedding_size])
        X_weighted_L2 = np.empty([len(nodes), embedding_size])

        for index in range(len(nodes)):
            item = nodes[index]
            index_one = item[0]
            index_two = item[1]
            vec_one = normalized_embedding[index_one]
            vec_two = normalized_embedding[index_two]


            #combine_vector = np.append(vec_one, vec_two, axis=0)
            X_averge[index, :] = np.add(vec_one, vec_two)/2.0
            X_hadamard[index, :] = np.multiply(vec_one, vec_two)

            X_weighted_L1[index, :] = np.absolute(np.subtract(vec_one, vec_two))#np.divide(vec_one, vec_two)  # averge
            X_weighted_L2[index, :] = np.square(np.subtract(vec_one, vec_two))#np.subtract(vec_one, vec_two) #
        X = [X_averge, X_hadamard, X_weighted_L1, X_weighted_L2]
        return X

    def run_LINE(self,  G, proximity, nodes_mapping,  edges_select_connected, edges_select_not_connected, embedding_dim, labels, task):
        args = data_obj(proximity, embedding_dim)
        data_loader = DBLPDataLoader(G)

        args.num_of_nodes = data_loader.num_of_nodes
        tf.reset_default_graph()
        model = LINEModel(args)
        sess = tf.Session()

        init = tf.global_variables_initializer()
        sess.run(init)

        learning_rate = model.learning_rate
        sampling_time, training_time = 0, 0

        u_i, u_j, label = data_loader.fetch_batch(batch_size=args.batch_size, K=args.K)
        pathModel = os.getcwd()  #

        embedding_path = os.path.join(pathModel, "line_model_3"+ proximity+ timestr) # 1 for facebook, 2 for the arX-py
        flag_get = exists(embedding_path)

        if not flag_get:
            with tf.Session() as sess:
                print(args)
                print('batches\tloss\tsampling time\ttraining_time\tdatetime')
                tf.global_variables_initializer().run()
                initial_embedding = sess.run(model.embedding)
                learning_rate = args.learning_rate
                sampling_time, training_time = 0, 0
                for b in range(args.num_batches):
                    t1 = time.time()
                    u_i, u_j, label = data_loader.fetch_batch(batch_size=args.batch_size, K=args.K)
                    feed_dict = {model.u_i: u_i, model.u_j: u_j, model.label: label, model.learning_rate: learning_rate}
                    t2 = time.time()
                    sampling_time += t2 - t1
                    if b % 100 != 0:
                        sess.run(model.train_op, feed_dict=feed_dict)
                        training_time += time.time() - t2
                        if learning_rate > args.learning_rate * 0.0001:
                            learning_rate = args.learning_rate * (1 - b / args.num_batches)
                        else:
                            learning_rate = args.learning_rate * 0.0001
                    else:
                        loss = sess.run(model.loss, feed_dict=feed_dict)
                        print('%d\t%f\t%0.2f\t%0.2f\t%s' % (b, loss, sampling_time, training_time,
                                                            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
                        sampling_time, training_time = 0, 0
                    if b % 1000 == 0 or b == (args.num_batches - 1):
                        embedding = sess.run(model.embedding)
                        normalized_embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
            save_picle(embedding_path, normalized_embedding)
        else:
            normalized_embedding = load_pickle(embedding_path)
        print normalized_embedding.shape
        # get the data
        if task =="multi_label_prediction":
            list_mico, list_maco =  self.mutli_lables(G, normalized_embedding, labels, nodes_mapping, \
                                                      embedding_size = 128, clf = LogisticRegression(), flag = "supervised")
            return list_mico, list_maco, embedding_path

        else:
            X_1_list = self.get_value(edges_select_connected, normalized_embedding)
            X_0_list = self.get_value(edges_select_not_connected, normalized_embedding)

            operation_get = ["average", "hadamard", "absolute value", "square of substract"]

            # print "the numpy array shape is:"
            # print X.shape
            # print y.shape
            roc_comparasion = []
            accuracy_comparasion = []
            print "########################"
            print "########################"
            print operation_get
            for i in range(len(X_1_list)):
                #print "operation are %s" %operation_get[i]
                X_1 = X_1_list[i]
                X_0 = X_0_list[i]
                X = np.append(X_0, X_1, axis = 0)
                y_0 = np.zeros(X_0.shape[0])
                y_1 = np.ones(X_1.shape[0])
                y = np.append(y_0, y_1, axis = 0)
                y = np.reshape(y, (y.shape[0], 1))

                X_shuffle, y_shuffle = shuffle(X, y)

                X_train, X_test, y_train, y_test = train_test_split(X_shuffle, y_shuffle, test_size=.5, random_state=0)
                #print y_score.round()
                clf = LogisticRegression()
                clf.fit(X_train, y_train)
                y_score = clf.predict_proba(X_test)[:,1]
                micro_f1 = f1_score(y_test, y_score.round(), average='micro')
                macro_f2 = f1_score(y_test, y_score.round(), average='macro')
                #
                fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score)
                roc_auc_value = metrics.auc(fpr, tpr)

                scores = cross_val_score(clf, X_shuffle, y_shuffle, cv=10)
                roc_comparasion.append(roc_auc_value)
                accuracy_comparasion.append(np.mean(scores))
            return roc_comparasion, accuracy_comparasion, embedding_path # return the auc value and 10 folders values

    def mutli_lables(self, G, normalized_embedding, labels, nodes_mapping,  embedding_size = 128, clf = LogisticRegression(),
                     flag = "supervised"):
        list_mico_lg, list_maco_lg = self.model_evaludate(G, normalized_embedding, labels, nodes_mapping, clf, embedding_size, flag)
        # list_mico_lg, list_maco_lg = self.model_evaludate(labels,embedding_size, clf, flag )
        return list_mico_lg, list_maco_lg

    def model_evaludate(self, G, normalized_embedding, labels,nodes_mapping ,clf, embedding_size = 128, flag = "supervised"):

        count = 0
        un_seen_node = 0
        print normalized_embedding.shape

        if nodes_mapping:
            print "the nodes mapping data"
            X = np.empty((len(nodes_mapping), embedding_size))
            for node in nodes_mapping:
                try:
                    vec_one = normalized_embedding[node]
                except:
                    vec_one = np.random.rand(embedding_size)
                    un_seen_node +=1
                X[node, :] = vec_one
            print "the unseen nodes %d" %un_seen_node
        else:
            a =  G.nodes()
            X = np.empty((len(a), embedding_size))
            for node in G.nodes():
                try:
                    vec_one = normalized_embedding[node]
                except:
                    vec_one = np.random.rand(embedding_size)
                    un_seen_node +=1
                X[node, :] = vec_one
            print "the unseen nodes %d" %un_seen_node

        list_mico = []
        list_maco = []
        # items = list(reversed([p/10.0 for p in range(1, 10)])) #list(reversed(list1))
        items = [p/10.0 for p in range(1, 10)]

        pathModel = os.getcwd()  #
        label_infor = pathModel + "blog_label.pickle"
        x_data = pathModel + "blog_data.pickle"
        print "save the data"
        print label_infor
        print x_data

        save_picle(label_infor, labels)
        save_picle(x_data, X)

        print "the labels"
        print labels
        print "the x is"
        print X
        print X.shape
        print labels.shape
        clf = LogisticRegression(solver ="sag")
        for item in items:
            print "the item is"
            print item
            X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size= item, random_state=51)
            if flag == "supervised":
                y_score = OneVsRestClassifier(clf).fit(X_train, y_train).predict(X_test)
            else:
                y_score = clf.fit(X_train).predict(X_test)
            # item_preict = []
            # for item_data in y_score:
            #     if item_data.any():
            #         item_preict.append(item_data)
            all_zeros = not np.any(y_score)
            micro_f1 = f1_score(y_test, y_score, average='micro')
            macro_f2 = f1_score(y_test, y_score, average='macro')

            print micro_f1
            print macro_f2

            list_mico.append(micro_f1)
            list_maco.append(macro_f2)
        print list_mico
        print list_maco
        return list_mico, list_maco
