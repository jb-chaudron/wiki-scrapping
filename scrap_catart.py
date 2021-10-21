#Importation des bibliothèque pertinentes

from pywikibot.data.api import PropertyGenerator, PageGenerator,ListGenerator, QueryGenerator,Request
import pywikibot as pw
import pandas as pd

#Definition des fonction importantes
def request_cat(cont=0):
    if isinstance(cont,int):
        pagegen = Request(site=site,parameters={"action":"query",
                                                         "generator":"allpages",
                                                         "gapfilterredir":"nonredirects",
                                                        "gaplimit" : 500
                                                        })
        out = pagegen.submit()
    else:
        pagegen = Request(site=site,parameters={"action":"query",
                                                         "generator":"allpages",
                                                         "gapfilterredir":"nonredirects",
                                                        "gaplimit" : 500,
                                                        "gapcontinue" : cont
                                                        })
        out = pagegen.submit()
    return out

def scrap_catart():

    cond = True
    req = request_cat()
    cont_param,titre = parse_req(req)
    scrap = []

    while cond:
        req = request_cat(cont_param)
        cont_param,titre = parse_req(req)

        scrap += titre
        print(titre[0])

    return scrap

def parse_req(req):
    cont_param = req["query-continue"]['allpages']["gapcontinue"]
    temp_req = req["query"]['pages']

    titre = [temp_req[i]["title"] for i in temp_req.keys()]

    return cont_param,titre

#Sélection du site
site = pw.Site("wikipedia:fr")

#Scraping et enregistrement des articles
art = scrap_catart()
df_art = pd.Series(art)
df_art.to_csv("ser_art.csv")
