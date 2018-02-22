# RANE

**RANE**: Relation-Aware Representation Learning in Information Networks

RANE is a representation learning approach that simultaneously learns multiple explicit as well as implicit relations.  Paper regarding with model details is currently under submission.  Please send us a request for the article access if interested.  We encourage non-commercial usage for research purpose. 

**Environment**: python 2.7
**Prerequisites**: gensim==3.2.0, Scikit-learn, networkx, pandas

**Basic Usage**

(1) **link prediction**
- *run* "RANE_multi_label_prediction.py" and adjust the data name to evaluate corresponding sub-task.

(2) **multi-label classification**
- *run* "RANE_multi_label_prediction.py" and adjust the data name to evaluate corresponding sub-task

(3) **node clustering**
- *train* RANE model on corresponding data
- *run* RANE_Calinski_Harabaz_score.py to get Calinski-Harabaz score and t-SNE visualizations

**Data**
As mentioned in paper, we currently maintain three five data-sets for model evaluation: Facebook, Arxiv, PPI, Wiki POS, Blog.

You can change the algorithm type, now this version code can be RANE or LINE algorithm.


For the implement of other comparison algorithms are below:

SDNE: https://github.com/suanrong/SDNE

Node2vec: https://github.com/aditya-grover/node2vec

DeepWalk: Set p =1 and q =1 by using node2vec code

If further any questions or suggestions, you are welcome to send email via: weizhili2014@gmail.com
