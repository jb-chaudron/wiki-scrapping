import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
#from node2vec import Node2Vec
import time
import csrgraph as cg
import nodevectors

def cooc(a):
  a = a.stack()
  a = a[a >= 1].rename_axis(('source', 'target')).reset_index(name='weight')

  return a

#Paramètres Node2Vec
dim = 1_000
w_len=16
n_walk = 5_000
time0 = time.time()
print("Début")
#path = "cat_wiki.csv"
#cat = pd.read_csv(path,index_col=0)
#cat = cat.T.dot(cat)

##rint("Matrice OK",time.time()-time0)
#new_coo = cooc(cat)
#G = nx.from_pandas_edgelist(new_coo,  edge_attr=True)
G = nx.read_adjlist("graph_propre.adjlist")
ggvec_model = nodevectors.GGVec(n_components=1_000)
embeddings = ggvec_model.fit_transform(G)
np.save("vecteur_model_1000dim",embeddings,allow_pickle=True)
ggvec_model.fit(G)
ggvec_model.save('model_1000dim')
ggvec_model.save_vectors("wiki_cat2vec.bin")

#embeddings.save_vectors("wiki_cat2vec.bin")

"""
print("Graph OK",time.time()-time0)
# Generate walks
node2vec = Node2Vec(G,
                    dimensions=dim,
                    walk_length=w_len,
                    num_walks=n_walk,
                    workers=10
                    )
print("Prepr node2vec OK",time.time()-time0)
# Learn embeddings
model = node2vec.fit(window=5,
                    min_count=1)
print("Node2vec fini !",time.time()-time0)
df_out = pd.DataFrame(0,
                      index=np.array([x for x in G.nodes]),
                      columns = ["dim : {}".format(j) for j in range(dim)])

for i in df_out.index:
  df_out.loc[i,:] = model.wv[i].astype(float)
print("Fini !",time.time()-time0)
df_out.to_csv("emb_cat.csv")
"""
