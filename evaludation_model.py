import numpy as np
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, cross_validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
from numpy import linalg as LA
import random
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_multilabel_classification
from sklearn.svm import SVC

from sklearn.metrics import f1_score

class mode_evaludatoin():
    def __init__(self, model, G, embedding_size = 128):
        self.model = model
        self.G = G
        self.embedding_size = embedding_size


    def clustering_embedding(self, labels, embedding_size):
        return


    def mutli_lables(self, labels, embedding_size = 128, clf = LogisticRegression(C= 1, penalty = "l2", tol=0.01),
                     flag = "supervised"):

        list_mico_lg, list_maco_lg = self.model_evaludate(labels,clf, embedding_size, flag)
        # list_mico_lg, list_maco_lg = self.model_evaludate(labels,embedding_size, clf, flag )
        return list_mico_lg, list_maco_lg

    def model_evaludate(self, labels,clf, embedding_size = 128, flag = "supervised"):
        G = self.G
        model = self.model
        a =  G.nodes()
        print "#######"
        print a
        X = np.empty((len(a), embedding_size))
        count = 0
        un_seen_node = 0
        for node in G.nodes():
            try:
                vec_one = model[str(node)]
            except:
                vec_one = np.random.rand(embedding_size)
                un_seen_node +=1

            X[count, :] = vec_one
            count +=1
        print "the unseen nodes %d" %un_seen_node
        list_mico = []
        list_maco = []
        items = [p/10.0 for p in range(1, 10)]
        print "the labels"
        print labels
        print "the x is"
        print X
        for item in items:
            X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size= item, random_state=51)
            if flag == "supervised":
                y_score = OneVsRestClassifier(clf).fit(X_train, y_train).predict(X_test)
            else:
                y_score = clf.fit(X_train).predict(X_test)
            item_preict = []
            for item in y_score:
                if item.any():
                    item_preict.append(item)
            all_zeros = not np.any(y_score)

            micro_f1 = f1_score(y_test, y_score, average='micro')
            macro_f2 = f1_score(y_test, y_score, average='macro')
            list_mico.append(micro_f1)
            list_maco.append(macro_f2)
        print list_mico
        print list_maco
        return list_mico, list_maco


    def auc_model(self, edges_select_connected, edges_select_not_connected):
        X_1_list = self.get_value(edges_select_connected)
        X_0_list = self.get_value(edges_select_not_connected)

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
            #y_test = np.reshape(y_test, (y_test.shape[0]))

            micro_f1 = f1_score(y_test, y_score.round(), average='micro')
            macro_f2 = f1_score(y_test, y_score.round(), average='macro')

            fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score)
            roc_auc_value = metrics.auc(fpr, tpr)

            scores = cross_val_score(clf, X_shuffle, y_shuffle, cv=10)
            roc_comparasion.append(roc_auc_value)
            accuracy_comparasion.append(np.mean(scores))
        return roc_comparasion, accuracy_comparasion # return the auc value and 10 folders values

    def get_value(self, nodes):
        model = self.model
        embedding_size = self.embedding_size
        X_averge = np.empty([len(nodes), embedding_size])
        X_hadamard = np.empty([len(nodes), embedding_size])
        X_weighted_L1 = np.empty([len(nodes), embedding_size])
        X_weighted_L2 = np.empty([len(nodes), embedding_size])

        for index in range(len(nodes)):
            item = nodes[index]
            try:
                vec_one = model[str(item[0])]
            except:
                vec_one = np.ones(embedding_size)
            try:
                vec_two = model[str(item[1])]
            except:
                vec_two = np.ones(embedding_size)

            X_averge[index, :] = np.add(vec_one, vec_two)/2.0
            X_hadamard[index, :] = np.multiply(vec_one, vec_two)

            X_weighted_L1[index, :] = np.absolute(np.subtract(vec_one, vec_two))#np.divide(vec_one, vec_two)  # averge
            X_weighted_L2[index, :] = np.square(np.subtract(vec_one, vec_two))#np.subtract(vec_one, vec_two) #
        X = [X_averge, X_hadamard, X_weighted_L1, X_weighted_L2]
        return X
