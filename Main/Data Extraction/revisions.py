import multiprocessing
import pickle
import os

import pywikibot as pw
import pandas as pd


global SITE
SITE = pw.Site("wikipedia:fr")

path_data = "/scratch/jchaudro/data/articles_diachronie"
if not os.path.exists(path_data):
    os.makedirs(path_data)


articles = pd.read_csv("/scratch/jchaudro/data/BA_AdQ.csv",index_col=0)

def get_text_oldvers(nom_article):
    print(nom_article)
    try:

        page = pw.Page(SITE,nom_article)

        revisions_article= [dict(revision) for revision in page.revisions(content=False)]
        texte_revision =  [page.getOldVersion(oldid = revision["revid"]) for revision in revisions_article]

        for e,revision in enumerate(revisions_article):
            revision["texte"] = [texte_revision[e]]

        with open(path_data+"/"+nom_article+".pkl", "wb") as fp:   #Pickling
            pickle.dump(revisions_article,fp)
        print("ok")
    except:
        pass

pool = multiprocessing.Pool(5)
pool.map(get_text_oldvers,[article[0] for article in articles.values])
pool.close()
