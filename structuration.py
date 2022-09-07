# Wiki Parsing
import mwparserfromhell as mw
import pywikibot as pw

# Utils
import concurrent
import datetime
import numpy as np
import os
import pickle
import re

from collections import Counter
from functools import reduce
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm
from os import listdir
from os.path import isfile, join

# NLP
import spacy
import textacy

## Spacy
from spacy.tokens import Token, Span, Doc
from spacy.matcher import PhraseMatcher, Matcher
from spacy.tokens import Span, SpanGroup, Token, Doc
from spacy.language import Language

## Textacy
from textacy import preprocessing




"""
---------- Metadata Extraction ----------
"""
def metadonnee_article(texte_in):
    # 1 - Segmentation du texte par "\n" pour isoler charque retour à la ligne

    working_text = texte_in.split("\n")
    features = {tag : [] for tag in ["catégories","portails","qual","vote","oldid","date"]}

    port = []
    # 2 - Récupération des catégories
    cat = []
    for ligne in working_text[::-1]:
        if ligne.find("Catégorie:") != -1:

            work_cat = ligne.split("Catégorie:")[1]
            cat += [work_cat.translate(str.maketrans("","","]"))]

        #elif np.any([tag in ligne for tag in ["vote","oldid","date"]]):
        elif "de qualité" in ligne:

            work_ligne = ligne.translate(str.maketrans("","","[]{}"))
            work_ligne = work_ligne.split("|")

            feat = {tag[:-1] : [wl.split(tag)[1] for wl in work_ligne if wl.find(tag) != -1] for tag in ["vote=","oldid=","date="]}
            features.update(feat)
            features["qual"] += [work_ligne[0]]
            #features if tag == "qual"
        elif ligne.find("Portail|") != -1:

            work_port = ligne.translate(str.maketrans("","","[]{}"))
            work_port = work_port.split("|")
            port += [work_port[1:]]

        elif ligne.lower().find("multi bandeau|") != -1:

            work_port = ligne.translate(str.maketrans("","","[]{}"))
            work_port = "".join(work_port.split("|")[1:])
            port += [work_port.split("Portail ")]

        elif "==" in ligne:

            break
        else:
            continue

    features["catégories"] = cat
    features["portails"] = port[0] if len(port) != 0 else []

    #3 - Récupération du nombre de template, liens etc...
    mw_parsed = mw.parse(texte_in)

    modules = []
    for node in mw_parsed.nodes:
        name = str(node.__class__).split(".")[-1][:-2]
        if name == "Template":
            modules += ["Template : "+str(node._name)]
        else:
            modules += [name]

    dic_node = Counter(modules)
    features.update(dic_node)

    return features

def get_metadata(article):
    meta = metadonnee_article(article["texte"][0])
    meta.update({key : article[key] for key in ["revid",
                                               "parentid",
                                               "user",
                                               "userid",
                                               "timestamp",
                                               "size",
                                               "comment",
                                               "parsedcomment",
                                               "tags",
                                               "minor",
                                               "userhidden",
                                               "commenthidden",
                                               "titre"]})

    meta["timestamp"] = str(meta["timestamp"])

    return meta

"""
---------- Mediawiki Processing ----------
"""

def analyse_node(node):
    """
        Fonction principale, elle détermine ce qui sera fait en fonction de la classe de l'objet
            1) Test si c'est un template
            2) Test si c'est un lien wiki
            3) Test si c'est du Text
            4) Autres cas (on ne retourne rien pour eux)
    """

    if node.__class__ == mw.nodes.template.Template:

        out = node.params

        if (len(out) > 5) or (len(out)==0):
            out = ""
        elif len(out) == 1:
            out = str(out[0])
        else:
            out = " ".join([str(x) for x in out])

    elif node.__class__ == mw.nodes.wikilink.Wikilink:
        out = str(node)

    elif node.__class__ == mw.nodes.text.Text:
        out = str(node.value)

        """
        if ("[" in out) or ("]" in out):
            out = ""
        """

    elif node.__class__ == mw.nodes.heading.Heading:

        out = "< {} >".format(node.level) + " {} ".format(node.title) + "< {} >".format(node.level)

    else:

        out = node

    return " "+str(out)

def preparation_texte(texte_in):

    texte_out = " < 2 > Introduction < 2 > "+texte_in
    return " ".join([" < p_deb > "+x+" < p_fin > " for x in texte_out.split("\n\n")])

def mw_parser(texte_in):
    #print("pre_parsing ok")
    texte_out = mw.parse(texte_in)
    #print("parsing 1 Ok")
    texte_out = mw.parse("".join([analyse_node(j) for j in texte_out.nodes])).strip_code()
    #print("parsing 2 Ok")
    return texte_out

"""
---------- Textacy Preprocessing Pipeline ----------
"""
preproc = preprocessing.make_pipeline(mw_parser,
                                      preprocessing.normalize.bullet_points,
                                      preprocessing.normalize.quotation_marks,
                                      lambda text : preprocessing.remove.brackets(text,only=["curly","square"]),
                                      preprocessing.remove.html_tags,
                                      preparation_texte,
                                      preprocessing.normalize.whitespace)

"""
---------- Spacy Custom Function ----------
"""
@Language.component("section")
def section_span(doc):
    matcher = Matcher(nlp.vocab)
    pattern = []

    for i in range(2,10):
        pattern += [[{"ORTH": "<"},{"ORTH": str(i)},{"ORTH":">"}]+
                [{"OP":"?"} for x in range(15)]+
                [{"ORTH": "<"},{"ORTH": str(i)},{"ORTH":">"}]]
    #print(pattern)
    matcher.add("section",pattern)


    matches = matcher(doc)

    spans_min = [doc[start+3:end-3] for _, start, end in matches]
    spans_max = [doc[start:end] for _, start, end in matches]

    section_head_min = spacy.util.filter_spans(spans_min)
    section_head_max = spacy.util.filter_spans(spans_max)

    for e,span in enumerate(section_head_min):
        try:
            span._.level = int(section_head_max[e].text.split("<")[1][1])-1
        except:
            span._.level = 0

    group = SpanGroup(doc, name="header", spans=section_head_min, attrs={"heading": "True"})
    doc.spans["header"] = group

    bad_token = list(set(reduce(lambda a,b : a+b, [[start+i for i in range(3)]+[end-i for i in range(4)] for _,start,end in matches])))
    for i in bad_token:
        if i >= len(doc):
            break
        else:
            doc[i]._.bad = True

    return doc

@Language.component("paragraphe")
def paragraphe_span(doc):
    matcher = Matcher(nlp.vocab)
    pattern = [{"ORTH" : "<"},{"ORTH":"p_deb"},{"ORTH":">"},             # Match "<p_deb>"
               {"ORTH" : {"NOT_IN" : ["p_deb","p_fin"]}, "OP" : "*"},    # Match tout le texte qui suit
               {"ORTH" : "<"},{"ORTH":"p_fin"},{"ORTH":">"}]# Match "<p_fin>"

    matcher.add("paragraphes",[pattern])

    matches = matcher(doc)
    spans = [doc[start+3:end-3] for _, start, end in matches]

    section_head = spacy.util.filter_spans(spans)

    group = SpanGroup(doc, name="paragraphes", spans=section_head, attrs={"paragraphe": "True"})
    doc.spans["paragraphes"] = group

    bad_token = list(set(reduce(lambda a,b : a+b, [[start+i for i in range(3)]+[end-i for i in range(4)] for _,start,end in matches])))
    for i in bad_token:
        if i >= len(doc):
            break
        else:
            doc[i]._.bad = True
    return doc

@Language.component("remove_bad")
def remove_bad(doc):
    matcher = Matcher(nlp.vocab)
    pattern = [{"_" : {"bad" : True}}]

    matcher.add("mauvais",[pattern])

    matches = matcher(doc)
    bads = np.array(sorted([start for _,start,end in matches]))

    n_remove = lambda i, bad : i-sum(bads<=i)

    span_head = [[n_remove(int(head.start),bads),1+n_remove(int(head.end),bads)] for head in doc.spans["header"]]
    levels_head = [span._.level for span in doc.spans["header"]]
    span_para = [[n_remove(int(para.start),bads),1+n_remove(int(para.end),bads)] for para in doc.spans["paragraphes"]]


    doc = Doc(nlp.vocab,
              words=[t.text for t in doc if not ((t._.bad) or (t.text in ["<",">","p_fin","p_deb"]))],
              spaces=[bool(t.whitespace_) for t in doc if not ((t._.bad) or (t.text in ["<",">","p_fin","p_deb"]))])

    span_head = [doc[start:end] for start,end in span_head]
    span_para = [doc[start:end] for start,end in span_para]

    group = SpanGroup(doc, name="header", spans=span_head, attrs={"heading": "True"})
    doc.spans["header"] = group
    for span,level in zip(doc.spans["header"],levels_head):
        span._.level = level

    group = SpanGroup(doc, name="paragraphes", spans=span_para, attrs={"paragraphe": "True"})
    doc.spans["paragraphes"] = group

    return doc

@Language.component("tag_sect")
def attribution_section(doc):
    sections = [[span.start,span.end,span._.level,span.text] for span in doc.spans["header"]]
    sections_span = []

    for sect in sections:
        start = sect[0]
        level = sect[2]
        end = [fin for deb,fin,lev,text in sections if ((deb > start) and (level <= lev))]
        if len(end) == 0:
            end = len(doc)
        else:
            end = end[0]

        sections_span += [[start,end]]
        doc[start:end]._.sections = sect[3]
        doc[start:end]._.level = sect[2]

    group = SpanGroup(doc,
                     name = "sections",
                     spans = [doc[start:end] for start,end in sections_span])

    doc.spans["sections"] = group

    return doc

"""
---------- Preparation Spacy ----------
"""

Span.set_extension("heading",default=False)
Span.set_extension("paragraphe",default=False)
Span.set_extension("level",default=None)
Span.set_extension("sections",default=None)

Token.set_extension("bad",default=False)

#Doc.set_extension("meta",default=None)


nlp = spacy.load("fr_core_news_lg")
nlp.add_pipe("section",name="parse_sect",first=True)
nlp.add_pipe("paragraphe",name="parse_para",after="parse_sect")
nlp.add_pipe("remove_bad",name="remove_bad",after="parse_para")
nlp.add_pipe("tag_sect",name="tag_sect",after="remove_bad")

"""
---------- Ouverture Articles ----------
"""

folder_path = "/scratch/jchaudro/data/articles_diachronie"
files = [join(folder_path, f) for f in listdir(folder_path) if isfile(join(folder_path, f))]

def open_textes(path):

    try:
        with open(path,"rb") as f:

            article_temp = pickle.load(f)
            titre = path.split('/')[-1][:-4]

            out = []
            for article in article_temp:
                article["titre"] = titre
                out += [article]

            print(np.all(['titre' in list(x.keys()) for x in out]))
            print(titre)

        return out

    except:
        print(path + " -- pass -- ")
        pass


with concurrent.futures.ProcessPoolExecutor() as executor:
    articles = executor.map(open_textes, files[30:32])

#articles = [open_textes(texte) for texte in files[30:32]]
path_full_articles = "/scratch/jchaudro/data/articles_full/"

if not os.path.exists(path_full_articles):
    os.makedirs(path_full_articles)

print("dump vrac")
articles = [x for x in tqdm(articles)]
del files

with open(path_full_articles+"articles_vrac.pkl", "wb") as fp:   #Pickling
    pickle.dump(articles,fp)

print("saved")

print("Fin des ouvertures et transformation de textes")
articles = [article for list_art in tqdm(articles) for article in list_art]
print("Fin de l'association en une liste")

print("début du dump")
path_full_articles = "/scratch/jchaudro/data/articles_full/"

if not os.path.exists(path_full_articles):
    os.makedirs(path_full_articles)

with open(path_full_articles+"tous_articles.pkl", "wb") as fp:   #Pickling
    pickle.dump(articles,fp)

print("Sauvegarde de tout les articles")


"""
---------- Création Objets ----------
"""
"""
print("Chargement données")

with open("/scratch/jchaudro/data/articles_full/tous_articles.pkl","rb") as file:
    articles = pickle.load(file)

print("donnée chargées")
"""
docs = []
meta = []

print("Spacy Pipeline")
def to_spacy_doc(texte):
    if "texte" in list(texte.keys()):
        try:
            #print(list(texte.keys()))
            #print(texte["titre"]+ " Preproc - Metadata - Essai")
            texte_clean = preproc(texte["texte"][0])
            #print(" Preprocessing OK ")
            meta = get_metadata(texte)

            print(texte["titre"]+ " Preproc - Metadata - Réussit")
            return (texte_clean,meta)

        except:
            #print(texte["titre"]+" -- Pass --")
            pass
    else:
        print("PAS DE TEXTE !!")
        pass

with concurrent.futures.ProcessPoolExecutor() as executor:
    data = executor.map(to_spacy_doc, articles)

data = [x for x in data if x != None]
del articles

print("récupération objet data")
data = [(doc,meta) for doc,meta in tqdm(data) if "titre" in list(meta.keys()) ]


#print(data)

print("To Textacy !")
"""
def textacy_doc(doc_in):

    try:
        print(len(doc_in[0]))
        print(doc_in[1]["titre"])
        return textacy.make_spacy_doc(lang=nlp,data= doc_in)
    except:
        print("Except ! ")
        pass

def main(data):
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        docs = executor.map(textacy_doc,data)
    return docs


print("killed ?")
    #print(docs)
if __name__ == '__main__':
    docs = main(data)
"""
docs = [textacy.make_spacy_doc(lang=nlp,data=doc) for doc in tqdm(data)]
del data

#print(docs[0],type(docs[0]),)
#print(data[0])
#docs = [textacy.make_spacy_doc(lang=nlp,data= (doc,meta)) for doc,meta in tqdm(data)]

print("Textacy Corpus")
corpus = textacy.Corpus(lang=nlp,data=docs)

"""
for doc, meta in  nlp.pipe(data, as_tuples=True,n_process=50,batch_size=50):
    print(meta)
    print(doc[:10])
    doc._.meta = meta
    docs.append(doc)
    print(meta["titre"]+" -- Fait -- ")


corpus = textacy.Corpus(lang=nlp,data=docs)
"""
print("\n Corpus -- Fait -- ")

path_out = "/scratch/jchaudro/data/corpus"
if not os.path.exists(path_out):
    os.makedirs(path_out)

corpus.save(path_out+"/BA_AdQ_all_vers.bin")
print("Sauvegardé !")
