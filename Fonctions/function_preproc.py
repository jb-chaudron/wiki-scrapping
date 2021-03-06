from pywikibot import Site, Page
import pywikibot.textlib as pwt
from functools import reduce
import pandas as pd
import numpy as np
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize


"""
========== Fonctions pour le 2nd Parsing
"""

def analyse_node(node):
    """
        Fonction principale, elle détermine ce qui sera fait en fonction de la classe de l'objet
            1) Test si c'est un template
            2) Test si c'est un lien wiki
            3) Test si c'est du Text
            4) Autres cas (on ne retourne rien pour eux)
    """

    if node.__class__ == mwparserfromhell.nodes.template.Template:
        #out = pars_temp(node)
        out = ""
    elif node.__class__ == mwparserfromhell.nodes.wikilink.Wikilink:
        out = pars_wiklink(str.lower(str(node)))
    elif node.__class__ == mwparserfromhell.nodes.text.Text:
        out = str(node.value)
        if ("[" in out) or ("]" in out):
            out = ""
    else:
        out = ""

    return str(out).lower()

def pars_temp(tex):
    """
        Décompose les modules "templates" pour en extraire les
        informations importantes
            1) tex.split('|') => Sépare le texte en fonction des '|'
            2) On ne récupère pas le premier élément ([1:])
            3) On refusionne les autres parties

        Etant donné que certains templates étaient des successions de textes
        inutiles d'un point de vue sémantique, cette fonction n'est plus trop utile

        Elle pourrait être intéressante plus tard, pour sélectionner un sous ensemble des
        templates (cf la liste des templates de wiki, que l'on pourrait utiliser) et les
        traiter suivant cela
    """

    return " ".join(tex.split("|")[1:])

def pars_wiklink(tex):
    """
        Décompose les liens intra wiki pour récupérer le texte important
            1) Si ".jpg" est dedans, il s'agit d'une image alors on ne prend
               pas en compte ce lien
            2) Si "|" est dans le texte du lien, alors on ne prend que le dernier
               élément listé
            3) Avec un regex, on récupère uniquement les chiffres, les lettres et
               les espaces, puis on les concatènes
    """

    # Str lower pour facilité la reconnaissance du jpg selon qu'il soit en majuscule ou non
    tex = str(tex).lower()

    # Test si jpg est dans le text du lien
    if bool("jpg" in tex):
        return ""

    # Test si "|" est présent, si oui on split et on récupère le dernier élément
    if "|" in tex:
        tex = tex.split("|")[-1]

    # On récupère uniquement les lettres, numéros et espaces dans le texte
    out = re.findall('[\w ]',tex)

    # On retourne la vesion concaténée
    return "".join(out)

def nettoyage(texte):
    """
        Fonction pour le nettoyage et la standardisation des textes
            1) Lower
            2) Remove punctuation
            3) Tokenization
            4) Stop Word Filtering
            5) Stemming
            6) POS Tagging
    """
    # 1) Lowering
    out = str(texte).lower()

    # 2) R

    return out

"""
========== Fonctions post Parsing
"""
def retir_parent(text,inp="(",out=")"):
    de_pa = [x for x,y in enumerate(text) if y == inp]
    fi_pa = [x for x,y in enumerate(text) if y == out]
    li_pa = search_al(calc_dist(de_pa,fi_pa))

    mask = [list(range(x,y+1)) for x,y in zip([de_pa[lip] for lip in li_pa[:,0]],[fi_pa[fip] for fip in li_pa[:,1]])]
    mask = reduce(lambda a,b: a+b, mask)
    mask = np.isin(np.arange(0,len(text)),mask,invert=True)

    return ''.join(np.array(list(text))[mask])
    #return np.array(de_pa[:-1]), np.array(fi_pa[1:])


def search_al(arr):

    df = pd.DataFrame(arr,index=range(arr.shape[0]),columns=range(arr.shape[1]))
    out = []
    li = set(df.index)
    col = set(df.columns)

    #Continue tant qu'il y a des lignes ou des colonnes
    while bool(len(li)*len(col)) :
        ndf = df.loc[list(li),list(col)]

        #min_s0 = max_ligne(ndf)

        for i in ndf.index:
            #Récupère le meilleur candidat en colonne pour cette ligne
            best = pand_min(ndf,i,tp="col")
            """
            if bool(np.random.binomial(1,0.05)):
                print("{} = {} ?".format(i,best))
                print("taille de la matrice, lignes : {}, colonnes : {} \n\n".format(len(li),len(col)))
            """
            #Si le meilleur candidat pour la colonne préféré est aussi la ligne en question
            if i == pand_min(ndf,best,tp="ind"):
                #On ajoute le couple à la liste de sortie
                out += [[i,best]]

                #On retire les deux candidat à la liste
                li.remove(i)
                col.remove(best)


    return np.array(out)

def calc_dist(a,b):
    #Produit une matrice avec len(a) lignes et len(b) colonnes
    arr_dist = np.zeros((len(a),len(b)))

    #Calcul la distance entre chaque parenthèse.
    #Si une parenthèse fermante arrive avant une parenthèse ouvrante, on lui compte une
    #Distance de 100K
    sub = lambda c,d: b[c]-a[d]
    for i,j in itr.product(range(len(a)),range(len(b))):
        arr_dist[i,j] = sub(j,i) if sub(j,i) > 0 else 100_000

    return arr_dist

def pand_min(data,ent,tp="col"):
    if tp == "col":
        return data.columns[data.loc[ent,:].rank() == data.loc[ent,:].rank().min()][0]
    elif tp == "ind":
        return data.index[data.loc[:,ent].rank() == data.loc[:,ent].rank().min()][0]
    else:
        print("PB")

"""
========== Classe pour structurer les données
"""

class Article(object):
    """

    """

    def __init__(self, nom):

        self.nom = nom
        self.SITE = pw.Site("wikipedia:fr")


    """
    ------ Section 1 : Ouverture & Parsing du texte ------
    """
    def ouverture(self):
        self.page = pw.Page(self.SITE,self.nom)
        self.texte = self.page.text
        self.parsed = pw.textlib.extract_sections(self.texte,site=self.SITE)

    def parse(self):
        node_analysis = lambda a : "".join([analyse_node(j) for j in mwparserfromhell.parse(a).nodes])
        self.sect = {i.title : node_analysis(i) for i in self.parsed.sections}
        self.texte = reduce(lambda a,b : " ".join([a,b]),[i[1] for i in self.sect.items()])


    """
    ------ Section 2 : Structuration en graphe ------
    """
    def graphe(self):
        lev = ["= {} = ".format(self.nom)]
        self.g = nx.Graph()
        self.g.add_nodes_from(lev+list(self.sect.keys()))

    def lien_graphe(self):
        """
            Fonction pour établir le graphe représentant
            la structure hierarchique des articles

            Rajouter un attribut sur le niveau hierarchique du noeud
        """
        # Fonction établissant le niveau hierarchique de la section
        trouve = lambda a : int(len(re.findall('[=]',a))/2)-1

        # Initialisation des variables
        lev = ["= {} = ".format(self.nom)]
        niv = 0
        last = lev[0]
        link = []
        self.g.nodes["= {} = ".format(self.nom)]["niveau"] = trouve("= {} = ".format(self.nom))
        self.maximum = 0

        # Boucle pour construire les liens
        for i in self.sect.keys():
            """
                2 tests
                    - On est à un niveau non répertorié (donc plus bas que ce qui a été vu)
                    - On est à un niveau hierarchique répertorié
            """
            # 1) On définit le niveau grâce à la fonction trouve
            niv_act = trouve(i)
            self.maximum = max([self.maximum,niv_act])

            self.g.nodes[i]["niveau"] = niv_act
            if len(lev)-1 < niv_act:
                # 2) Si le niveau est supérieur alors on en créer un nouveau
                lev += [i]
                link += [(i,lev[niv_act-1])]

            else :
                # 3) Si le niveau est inférieur ou égale, on change la section
                #    de niveau actuel, et on fait un lien avec la section supérieur
                lev[niv_act] = i
                link += [(i,lev[niv_act-1])]

        self.g.add_edges_from(link)

    """
    ------ Section 3 : Extraction des différentes dimensions d'intérêt ------
    """

    def get_phrase(self):
        self.sect_phrase = {i[0] : sent_tokenize(i[1]) for i in self.sect.items()}
        self.nb_ph = {i[0] : len(i[1]) for i in self.sect_phrase.items()}
        self.lg_ph = {i[0] : [len(j) for j in i[1]] for i in self.sect_phrase.items()}

    def get_mot(self):
        self.mots = word_tokenize(reduce(lambda a,b: " ".join([a,b]), [x[1] for x in self.sect.items()]))
        self.tok, self.nb_tok = np.unique(self.mots,return_counts=True)

    def get_verb(self):
        nlp = spacy.load("fr_core_news_sm")
        get_dim = lambda a : [a.morph.get(x) for x in ["Gender","Number","VerbForm","Voice"]]
        doc = nlp(reduce(lambda a,b: " ".join([a,b]), [x[1] for x in self.sect.items()]))
        self.verb = {i[0] : [get_dim(x) for x in nlp(i[1]) if x.pos_ == "VERB"] for i in self.sect.items()}

    def get_sect(self):
        self.nb_sect = {i : len([j for j in self.g.nodes if self.g.nodes[j]["niveau"] == i]) for i in range(self.maximum+1)}
