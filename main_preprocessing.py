from pywikibot import Site, Page
import pywikibot.textlib as pwt
from functools import reduce
import pandas as pd
import numpy as np
import function_preproc as fct


"""
    
"""
#Définition du site sur lequel on travail
SITE = Site("wikipedia:fr")

# Récupération de la page
p = Page(SITE,"Lyon")
te = p.text

# 1ère phase de parsing
pw_pars = pwt.extract_sections(te,site=SITE)

# 2eme phase de parsing
text_sect = {i.title : "".join([analyse_node(j) for j in mwparserfromhell.parse(i).nodes]) for i in pw_pars.sections}


pre = pwt.removeCategoryLinksAndSeparator(te,site=SITE)
sec = pwt.removeDisabledParts(pre,site=SITE)
ter = pwt.removeHTMLParts(sec)
qua = pwt.removeLanguageLinksAndSeparator(ter,site=SITE)
