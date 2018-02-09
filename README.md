# metanode2vec
For the paper submission, KDD, 2018: Learning Network Embedding with Implicit Relations

Package needed:
gensim, 3.2.0
Scikit-learn
networkx
pandas

(1). Run the metanode2vec.py directly. It will run the multi-labels prediction tasks. 
generate the embedding file Name is: methnodes_model_02_09_2018_update_2.txt, lINE 268.

Change it when train a new model. 

(2). For link prediction, and nodes, removing the 50% links and run metanode2vec.py 

(3). For the nodes clustering, run metanode2vec.py to get the model, then run the Calinski_Harabaz_score.py. 

