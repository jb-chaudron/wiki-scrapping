import pandas as pd
import random
import pywikibot as pw
from functools import reduce
import numpy as np
from scipy.spatial.distance import pdist,squareform
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import networkx as nx
from node2vec import Node2Vec
import multiprocessing

#Def des fonctions
class Dataclus(object):
    """
        1 - Créer un Graphe
            a. Récupère les noms des articles et des catégories
            b. Utilise les noms comme noeuds
            c. Associe une catégorie ou un article à une autre catégorie, si ils lui appartiennent
        2 - Node Embedding
            a. Effectue une marche aléatoire sur le graphe
            b. Applique Word2Vec sur les "phrases" créer de cette façon
        3 - Hyper-parameter Tuning HClustering
            a. Créer un HClustering avec n clusters
            b. Test le fit de k HCluster, ayant de n à n+k clusters
            c. Choisit le nombre de cluster maximisant (ou minimisant) le mieux 3 métriques
            d. Recommence le processus en diminuant l'écart du nombre de clusters
            e. renvoie le meilleur paramètre
        4 - Clustering final
            a. Fit le clustering optimale des données
            b. Renvoie les labels des articles et les dimensions de l'embedding
    """

    def __init__(self, d_art,d_cat):

        self.d_art = d_art
        self.d_cat = d_cat
        self.site = pw.Site("wikipedia:fr")
        self.it = 0

    """
        Partie 1) - Graphes et cie
    """
    def gen_graph(self):
        self.g = nx.Graph()
        #self.noeud = random.sample([x for x in set(list(self.d_art.articles)+list(self.d_cat.categorie))],k=400_000)
        self.noeud = [x for x in set(list(self.d_art.articles)+list(self.d_cat.categorie))]
        self.g.add_nodes_from(self.noeud)
        pool = multiprocessing.Pool(8)
        pool.map(self.get_lien,self.noeud)

    def get_lien(self,name):

        self.it += 1

        if bool(np.random.binomial(1,0.1)):
            print(100*self.it/len(self.noeud))
        else:
            pass

        p = pw.page.BasePage(self.site,name)
        if p.isRedirectPage() or p.isCategoryRedirect():
            pass
        elif p.is_categorypage():
            cat = pw.Category(self.site,name)
            out = [x.title() for x in cat.subcategories()]+[x.title() for x in cat.categories()]
        else:
            out = [x.title() for x in p.categories()]
            #+[x.title() for x in p.linkedPages()]
            self.g.add_edges_from([(name,x) for x in out])

    """
        Partie 2) - Embedding et cie
    """
    def embedd_nodes(self,n_dim=40,walk=16,n_walk=100):
        node2vec = Node2Vec(self.g, dimensions=n_dim, walk_length=walk, num_walks=n_walk)
        self.model = node2vec.fit(window=10, min_count=1)
        dat = self.model.wv.vectors[[True if x in set(self.d_art.articles) else False for x in self.g.nodes]]
        self.data = pd.DataFrame(dat,index=[x for x in self.g.nodes if x in self.d_art.articles])


    """
        Partie 3) - Hyperparameter Tuning et cie
    """
    def h_clus(self,deb,fin,pas):
        scor = np.zeros((pas,3))
        for j,i in enumerate(tqdm(np.linspace(deb,fin,pas))):
            i = int(i)
            ac = AgglomerativeClustering(n_clusters=i)
            ac.fit(self.data)
            scor[j,:] = [silhouette_score(self.data,ac.labels_),
                        calinski_harabasz_score(self.data,ac.labels_),
                        -davies_bouldin_score(self.data,ac.labels_)]

            return scor,[int(x) for x in np.linspace(deb,fin,pas)]

    def find_max(self,sc,n_clus):
        t_rank = np.argsort(np.argsort(sc,axis=0),axis=0).mean(axis=1)
        m_clus = n_clus[np.argsort(t_rank)[-1]]
        p = round(np.diff(n_clus).mean())
        return (m_clus-p/2,m_clus+p/2)

    def iterclus(self,d=2,f=5_000,p=10):

        while ((f-d)/p)>1:
            sc,n_clu = h_clus(self.data,d,f,p)
            d,f = find_max(sc,n_clu)
            print(d,f)
            print(sc)

        self.best_nclus = int(d)+1


    """
        Partie 4) - Extraction des données
    """
    def best_lab(self):
        ac = AgglomerativeClustering(n_clusters=self.best_nclus)
        ac.fit(self.data)

        self.lab = pd.Series(ac.labels_,index=self.data.index)

        self.lab.to_csv("cluster_labels.csv")
        self.data.to_csv("art_embedding.csv")


#Ouverture des fichiers
df_categorie = pd.read_csv("~/wiki_clust/data_ress/df_categorie.csv")
df_article = pd.read_csv("~/wiki_clust/data_ress/ser_art.csv",index_col=0)
df_article.columns = ["articles"]

# Lancement du Pipeline
## 1 - Création de l'objet
clu = Dataclus(df_article,df_categorie)
## 2 - Création du graphe
clu.gen_graph()
## 3 - Embedding
clu.embedd_nodes()
## 4 - Hyperparameter Tuning
clu.iterclus()
## 5 - Récupération des données
clu.best_lab()
pickle.dump(clu,open( "Clus_class.p", "wb" ))
