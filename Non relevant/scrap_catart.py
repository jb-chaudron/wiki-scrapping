#Importation des bibliothèque pertinentes

from pywikibot.data.api import PropertyGenerator, PageGenerator,ListGenerator, QueryGenerator,Request
import pywikibot as pw
import pandas as pd

#Definition des fonction importantes
def request_cat(cont=0):
    """
        A. Fonction qui itère sur l'ensemble des articles
            - cont : permet de spécifier le dernier articles ayant été "fetched"
    """
    
    #Vérifie si "cont" est un int ou bien une string, dans le 2nd cas, il faut reprendre le scrapping depuis "cont"
    if isinstance(cont,int):
        #Produce a generator of pages, with a limit of 500 pages top
        pagegen = Request(site=site,parameters={"action":"query",
                                                         "generator":"allpages",
                                                         "gapfilterredir":"nonredirects",
                                                        "gaplimit" : 500
                                                        })
        out = pagegen.submit()
        
    else:
        #Same thing as above, but strat the iteration at the article saved in "cont" variable
        pagegen = Request(site=site,parameters={"action":"query",
                                                         "generator":"allpages",
                                                         "gapfilterredir":"nonredirects",
                                                        "gaplimit" : 500,
                                                        "gapcontinue" : cont
                                                        })
        out = pagegen.submit()
        
    #Return the generator produced by the function
    return out


def parse_req(req):
    """
        Aim : Parsing the query made through the generator produced via the "request_cat" function
            1) Assess if this is the last query, i.e. that every article have been fetched
                Yes : Return a cond=False, that will stop the while loop
                No : Parse the query to find the continue article (the one that will be used for the next iteration)
                     Parse the query to extract every article's title returned by the generator
    """
    
    #Assess if this is the last generator, if not, return the title of the article to be used for the next generator's start
    if not "query-continue" in req.keys():
        cond = False
        cont_param = None
    else:
        cont_param = req["query-continue"]['allpages']["gapcontinue"]
        cond=True
       
    #Return the list of all wikipedia article queried by the generator
    if "pages" in req["query"].keys():
        temp_req = req["query"]['pages']
        titre = [temp_req[i]["title"] for i in temp_req.keys()]
    else:
        titre = []


    return cont_param,titre,cond

def scrap_catart():
    """
       Loop over all the article in wikipedia to find their names 
    """
    
    #First iteration
    cond = True
    req = request_cat()
    cont_param,titre,cond = parse_req(req)
    scrap = []

    #Loop until every articles have been found
    while cond:
        req = request_cat(cont_param)
        cont_param,titre,cond = parse_req(req)

        scrap += titre
        print(cont_param)

    #Return the name of the articles queried by the procedure
    return scrap

#Select the site
site = pw.Site("wikipedia:fr")

#Scraping et enregistrement des articles
art = scrap_catart()
df_art = pd.Series(art)
df_art.to_csv("ser_art.csv")
