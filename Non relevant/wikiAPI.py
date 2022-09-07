import requests
S = rq.Session()


class page:
    def __init__(self,nom,parents,enfants):
        self.nom = nom
        self.parents = parents
        self.nb_par = len(np.unique(parents))
        self.enfants = enfants
        self.nb_enf = len(np.unique(enfants))

    def scrap_vers():
        pass

    def scrap_user():
        pass



def ouvre_cat(prem_cat=0,
              url="https://fr.wikipedia.org/w/api.php",
              S=S,
              min_link= 100,
              limite=500
              ):
    """
        Fonction pour récupérer N Catégories
    """
    if prem_cat == 0:
        PARAMS = {
        "action": "query",
        "format": "json",
        "list": "allcategories",
        "acmin":min_link,
        "aclimit": limite
        }
    else:
        PARAMS = {
        "action": "query",
        "format": "json",
        "list": "allcategories",
        "acfrom": prem_cat,
        "acmin":min_link,
        "aclimit": limite
        }

    R = S.get(url=url, params=PARAMS)
    DATA = R.json()
    print(DATA["query"].keys())
    CATEGORIES = DATA["query"]["allcategories"]

    return [x["*"] for x in CATEGORIES]


def scrap_cat(len_scap=500):
    """
        Fonction pour récupérer toutes les catégories qui
        nous intéressent.
    """
    #Initialise l'array qui va garder les données
    qt = np.array([],dtype="str")

    #Ouvre len_scap catégories et les envoies dans qt
    cat = ouvre_cat(0,url,S,limite=len_scap)
    qt = np.append(qt,cat[:-1])

    #Répète l'opération, jusqu'à temps qu'on ait épuisé les catégories
    ini = len_scap
    while ini == len_scap:
        cat = ouvre_cat(qt[-1],url,S,limite=len_scap)
        qt = np.append(qt,cat[:-1])

        ini = len(cat)

    return qt
