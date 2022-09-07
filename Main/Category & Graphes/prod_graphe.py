import networkx as nx
from functools import reduce, partial
import multiprocessing
import pywikibot as pw
import itertools
from functools import reduce
from tqdm import tqdm
import numpy as np
import time

def fct_cat():
    cond = lambda a : a.isEmptyCategory() or a.isRedirectPage() or a.isCategoryRedirect()
    return [x for x in a.subcategories() if not (x in fait) or not (cond(x))]
"""
def get_cat(list_in,fait):

    pool = multiprocessing.Pool(10)
    out = pool.map(fct_cat,list_in)
    pool.terminate()

    return out
"""
name = np.load("noms.npy")
SITE = pw.Site("wikipedia:fr")
global t0
global it
global fait
t0 = time.time()

def fct_multi(inp):
    if np.random.binomial(1,0.1):
        print("Multi",t0-time.time())

    return set([x for x in pw.Page(SITE,inp).categories()])
try:
    cat_inpset = np.load("cat_qual.npy")
except:
    pool = multiprocessing.Pool(6)

    cat_inp = pool.map(fct_multi,name)
    pool.terminate()
    cat_inpset = np.array(list(reduce(lambda a,b : a.union(b),cat_inp)))
    np.save("cat_qual",cat_inpset,allow_pickle=True)

print(cat_inpset)
#Init graphe & Noeud d'amorçe
G = nx.Graph()
afaire = []
encours = cat_inpset
fait = []
t0 = time.time()
it = 0
def pseudo_fct(inp):
    if np.random.binomial(1,0.1):
        print(t0-time.time()," : ",it)
    return [x for x in inp.categories() if not x.title() in fait]

while len(encours) > 0 :

    print(len(fait))
    print(len(encours))

    # Récupère les pages suivantes et leur noms, et les liens à ajouter
    #cond = lambda a : a.isEmptyCategory() or a.isRedirectPage() or a.isCategoryRedirect()
    #fct_in = lambda a :
    t0 = time.time()
    print("{} Commence".format(it))
    pool = multiprocessing.Pool(10)
    new_cat = pool.map(pseudo_fct,encours)
    pool.terminate()
    print("{} finit".format(it))
    #[[x for x in y.categories()] for y in tqdm(encours)]
    #name_cat = reduce(lambda a,b: a.union(b),[set([b.title() for b in a]) for a in new_cat])

    # Mise à jour de la BL et du graphe
    #print([x.title() for x in l_in])
    print("Ajout des faits")

    enc_titre = [x.title() for x in encours]
    fait += enc_titre

    #G.add_nodes_from([x.title() for x in l_in])

    # Récupération des liens
    print("Ajout des liens etc...")
    liens = reduce(lambda a,b : a+b, [[(z.title(),y.title()) for y in x] for x,z in zip(new_cat,encours)])
    G.add_edges_from([(link[0],link[1]) for link in liens])

    print("Nouveau à faire")
    #enc_titre = [x.title() for x in encours]
    encours = np.unique([page for ss_list in new_cat for page in ss_list])
    #np.unique(np.array(new_cat,dtype=object).flatten())
    encours = [x for x in encours if (not x.title() in fait)]
    #encours = [x for x in reduce(lambda a,b : a.union(b), [set([y.title() for y in x]) for x in new_cat]) if (not x in fait)]
    print("Nombre de catégories {}".format(len(encours)))
    it += 1

nx.write_adjlist(G,"Graph_classes.adjlist")
print("terminado !")
