import pywikibot as pw
import pywikibot.textlib as pwt
import multiprocessing
import function_preproc as fct
from tqdm import tqdm
import nltk
from functools import reduce
import pandas as pd

def concatenation(dfs):
    colonnes = reduce(lambda a,b: a.union(b), [set(x[1].columns) for x in dfs.items()])
    indexes = reduce(lambda a,b: a+b, [[x[0]] for x in dfs.items()])

    n_df = pd.DataFrame(0,columns=list(colonnes),index=indexes)

    for i in dfs.keys():
        n_df.loc[i,dfs[i].columns] = [x for x in dfs[i].iloc[0,:]]

    return n_df

def recup_data(obj):
    obj.ouverture()
    cont = obj.parse()
    if cont:
        cont = obj.nettoyage()
        if cont :
            obj.graphe()
            obj.lien_graphe()
            obj.get_sect()
        else:
            pass
    else:
        pass


# 1 - Ouverture etc...
SITE = pw.Site("wikipedia:fr")
p = pw.Page(SITE,'Wikipédia:Articles vitaux')
trc = [i for i in p.linkedPages()]
data = {i.title() : fct.Article(i.title()) for i in trc}

# 2 - Traitement des données
nltk.download('punkt')
nltk.download("stopwords")

"""
pool = multiprocessing.Pool(4)
pool.map(lambda a : recup_data(a[1]),list(data.items()))
"""

for i in tqdm(list(data.keys())):
    recup_data(data[i])

# 3 - Structuration des données
dic_df = {x[0] : x[1].data for x in data.items()}
df_out = concatenation(dic_df)

# 4 - Exportation du CSV
df_out.to_csv("data_v1.csv")
