# metanode2vec
For the paper submission, KDD, 2018: Relation-Aware Representation Learning in Information Networks

Environment: python 2.7

Package needed:
gensim==3.2.0,
Scikit-learn,
networkx,
pandas

Three tasks:

(1). Link prediction:

run "RANE_multi_label_prediction.py" directly. Change the dataset through the data_set

There are three data-sets in submission: Facebook, Arxiv, PPI.


(2). multi-labels classification:

run "RANE_multi_label_prediction.py" directly. Change the dataset through the data_set

There are three data-sets in submission: PPI, wiki pos, Blog.

Because the node label (index) of Blog is not sequence, using the "blog_data_evaluation.py" to do the evaluation.

(3). nodes clustering

After getting the model, using "RANE_Calinski_Harabaz_score.py" to get the Calinski Harabaz score and TSNE data visualization

If you have any questions, please contact me with email: weizhili2014@gmail.com