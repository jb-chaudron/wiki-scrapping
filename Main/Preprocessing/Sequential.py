# Wiki Parsing
import mwparserfromhell as mw
import pywikibot as pw

# Utils
import concurrent
import datetime
import numpy as np
import os
import pandas as pd
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

n_work = 5

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
            out = " ".join([str(x) for x in out]).translate(str.maketrans("","","[]{}"))

    elif node.__class__ == mw.nodes.wikilink.Wikilink:
        if node.text == None:
            out = str(node)
        elif "=" in node.text:
            #print(str(node))
            out = clean_temp(" ".join(node.text.split("=")[1:]))
            if "|" in out:
                out = ".".join(out.split("|")).translate(str.maketrans("","","[]{}"))
        else:
            out = node


    elif node.__class__ == mw.nodes.text.Text:
        out = str(node.value)

        """
        if ("[" in out) or ("]" in out):
            out = ""
        """

    elif node.__class__ == mw.nodes.heading.Heading:

        out = "< {} >".format(node.level) + " {} ".format(node.title) + "< {} >".format(node.level)

    #elif "tag" in str(node.__class__):


    else:

        out = node

    return " "+str(out)

def clean_temp(str_in):
    str_in = str_in.replace(" =","=")
    #print(str_in)
    space = str_in.split(" ")
    #print(space)
    egal = [". "+" ".join(x.split("=")[1:]) if "=" in x else x for x in space]
    #print(egal)
    if len(egal) == 0 :
        return " "
    elif len(egal) == 1:
        return egal[0]
    else:
        return " ".join(egal)

def preparation_texte(texte_in):
    #print("preparation OK")
    texte_out = " < 2 > Introduction < 2 > "+texte_in
    return " ".join([" < p_deb > "+x+" < p_fin > " for x in texte_out.split("\n\n")])

def mw_parser(texte_in):
    #print("pre_parsing ok")
    texte_out = mw.parse(texte_in)
    #print("parsing 1 Ok")
    texte_out = mw.parse("".join([analyse_node(j) for j in texte_out.nodes])).strip_code().strip()
    #print("parsing 2 Ok")
    texte_out = texte_out

    return texte_out

"""
---------- Textacy Preprocessing Pipeline ----------
"""
preproc = preprocessing.make_pipeline(mw_parser,
                                      preprocessing.normalize.bullet_points,
                                      preprocessing.normalize.quotation_marks,
                                      lambda text : text.translate(str.maketrans("","","[]{}")),
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

folder_path = "/scratch/jchaudro/data/articles_diachronie/"
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

            #print(np.all(['titre' in list(x.keys()) for x in out]))
            #print(titre)

        return out

    except:
        print(path + " -- pass -- ")
        pass

"""
---------- Nettoyage & Preprocessing
"""

def to_spacy_doc(texte):
    if "texte" in list(texte.keys()):
        try:
            #print(list(texte.keys()))
            #print(texte["titre"]+ " Preproc - Metadata - Essai")
            texte_clean = preproc(texte["texte"][0])
            #print(" Preprocessing OK ")
            meta = get_metadata(texte)

            #print(texte["titre"]+ " Preproc - Metadata - Réussit")
            return (texte_clean,meta)

        except:
            #print(texte["titre"]+" -- Pass --")
            pass
    else:
        print("PAS DE TEXTE !!")
        pass

def textacy_doc(doc_in):

    try:
        #print(len(doc_in[0]))
        #print(doc_in[1]["titre"])
        return textacy.make_spacy_doc(lang=nlp,data= doc_in)
    except:
        print("Except ! ")
        pass

def traitement_sequentiel(file,meta=True):

    #print("Hello !!")
    with open(file,"rb") as f:
        print(f)
        article_temp = pickle.load(f)
        titre = file.split('/')[-1][:-4]

        print("--- Open : "+titre+" ---")
        if os.path.exists("/scratch/jchaudro/data/corpus/"+titre+".bin"):
            return None
        elif not os.path.exists("/scratch/jchaudro/data/clean_meta/"+titre+"pkl") :
            out = []
            for article in article_temp:
                article["titre"] = titre
                out += [article]
        elif os.path.exists("/scratch/jchaudro/data/clean_meta/"+titre+"pkl"):
            with open("/scratch/jchaudro/data/clean_meta/"+titre+"pkl","rb") as f:
                out = pickle.load(f)
                nlp.max_length = max([len(doc) for doc,meta for doc in out]) + 100

                print("--- To_textacy ! : "+titre+" ---")
                out = [textacy_doc((doc,meta)) for doc,meta in out]

                print("--- Save Corpus : "+titre+" ---")
                path_textacy = "/scratch/jchaudro/data/corpus/"
                if not os.path.exists(path_textacy):
                    os.makedirs(path_textacy)
                    corpus = textacy.Corpus(lang=nlp,data=out)
                    corpus.save(path_textacy+titre+".bin")

            return None
        else:
            print("--- Problème ? ---")
            print(file)
            return None

        print("--- Dump : "+titre+" ---")
        path_raw = "/scratch/jchaudro/data/articles_full/"
        with open(path_raw+titre+".pkl", "wb") as fp:   #Pickling
            pickle.dump(out,fp)

        print("--- To_spacy : "+titre+" ---")
        nlp.max_length = max([len(article) for article in out])
        out = [to_spacy_doc(article) for article in out]

        path_clean_meta = "/scratch/jchaudro/data/clean_meta/"
        if not os.path.exists(path_clean_meta):
            os.makedirs(path_clean_meta)

        with open(path_clean_meta+titre+".pkl", "wb") as fp:   #Pickling
            pickle.dump(out,fp)

        print("--- To_textacy ! : "+titre+" ---")
        out = [textacy_doc((doc,meta)) for doc,meta in out]

        print("--- Save Corpus : "+titre+" ---")
        path_textacy = "/scratch/jchaudro/data/corpus/"
        if not os.path.exists(path_textacy):
            os.makedirs(path_textacy)
        corpus = textacy.Corpus(lang=nlp,data=out)
        corpus.save(path_textacy+titre+".bin")

        return None

def try_except(files):
    try:

        traitement_sequentiel(files)
        return None
    except:
        return None

print(len(files))

def sort_files(files):
    return sorted(files,key=lambda a : os.path.getsize(a))

files = sort_files(files)

with tqdm(concurrent.futures.ProcessPoolExecutor(max_workers=n_work),max=len(files)) as executor:
    articles = executor.map(try_except, files,chunksize=100)
