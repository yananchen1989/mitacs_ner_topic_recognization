#############  ########## CUDA_VISIBLE_DEVICES
import pandas as pd 
import json,random

import datasets
ds_nerd = datasets.load_dataset('dfki-nlp/few-nerd', "supervised", cache_dir='/scratch/w/wluyliu/yananc/cache')
ds_notes = datasets.load_dataset('conll2012_ontonotesv5', "english_v12", cache_dir='/scratch/w/wluyliu/yananc/cache')
ds_conll = datasets.load_dataset('conll2003', cache_dir='/scratch/w/wluyliu/yananc/cache')

raw_datasets = ds_nerd

ds_nerd['train']['ner_tags'][35]






import spacy
ner_spacy_model = spacy.load('en_core_web_lg', disable=["tok2vec", "tagger", "attribute_ruler", "lemmatizer"])


import requests
import re
from collections import Counter

content = '''
The West lost self-confidence — and both Russian and Chinese leaders rubbed it in, putting out the word that these chaotic democratic systems were a spent force.
And then a totally unexpected thing happened: Russia and China each overreached.
Vladimir Putin invaded Ukraine and, to his surprise, invited an indirect war with NATO and the West. China insisted that it was smart enough to have its own local solution to a pandemic, leaving millions of Chinese underprotected or unprotected and, in effect,
'''

article = ner_spacy_model(content)
labels = [x.label_ for x in article.ents]
Counter(labels)


sentences = [x for x in article.sents]



texts = [
    "Net income was $9.4 million compared to the prior year of $2.7 million.",
    "Revenue exceeded twelve billion dollars, with a loss of $1b.",
]

nlp = spacy.load("en_core_web_sm")
for doc in nlp.pipe(texts, disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"]):
    # Do something with the doc here
    print([(ent.text, ent.label_) for ent in doc.ents])



from spacy import displacy
displacy.render(ner_spacy_model(content.strip()), jupyter=True, style='ent')

text = "When Sebastian Thrun started working on self-driving cars at Google in 2007, few people outside of the company took him seriously."

nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
html = displacy.render(doc, style="ent", page=True)


with open("ner_spacy_test.html", "w") as ff:
    ff.write(html+'------------\n') 












