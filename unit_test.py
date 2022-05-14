#############  ########## CUDA_VISIBLE_DEVICES
import pandas as pd 
import json,random
import numpy as np 
import datasets
ds_nerd = datasets.load_dataset('dfki-nlp/few-nerd', "supervised", cache_dir='/scratch/w/wluyliu/yananc/cache')
ds_notes = datasets.load_dataset('conll2012_ontonotesv5', "english_v12", cache_dir='/scratch/w/wluyliu/yananc/cache')
ds_conll = datasets.load_dataset('conll2003', cache_dir='/scratch/w/wluyliu/yananc/cache')


ds_nerd['train']['ner_tags'][35]



tag_map = {}
with open("/home/w/wluyliu/yananc/nlp4quantumpapers/utils/few_nerd_tag_map.tsv", 'r') as f:
    for line in f:
        tokens = line.strip().split('\t')
        tag = tokens[0]
        tag_tokens = tokens[1].split('-')
        if tag_tokens == ['O']:
            continue
        if tag_tokens[1] == 'other':
            tags_ = tag_tokens[0]
        else:
            tags_ = tag_tokens[1]
        tag_map[tag] = tags_



def sep_trunk(df_tmp):
    results = []
    for tag in df_tmp['tags'].unique():
        if tag == 'O':
            continue 

        df_tmp_f = df_tmp.loc[df_tmp['tags']==tag]

        list_of_df = [d for _, d in df_tmp_f.groupby(df_tmp_f.index - np.arange(len(df_tmp_f)))]
        mentions = [' '.join(df_tag['tokens'].tolist()) for df_tag in list_of_df]
        results.append((' <=> '.join(mentions), tag_map[tag]))
    return results



def get_ft(dsn):
    path = "/scratch/w/wluyliu/yananc/few_nerd_supervised"
    infos = []
    buffer = []
    with open("{}/{}.txt".format(path, dsn), 'r') as f:
        for line in f:
            if line =='\n':
                
                tokens = []
                tags = []
                ents = {}

                for j in buffer:
                    token = j.split('\t')[0]
                    tag = j.split('\t')[1]
                    tokens.append(token)
                    tags.append(tag)
            
                sent = ' '.join(tokens)

                buffer = []
                # print('----------------------------------')
                df_tmp = pd.DataFrame(zip(tokens, tags), columns=['tokens','tags'])

                ner_tag = sep_trunk(df_tmp)
                for ii in ner_tag:
                    infos.append((sent + " The {} is".format(ii[1]), ii[0]))
                
            else:
                buffer.append(line.strip())
    df_ft = pd.DataFrame(infos, columns=['text1', 'text2'])
    df_ft.drop_duplicates(inplace=True)
    return df_ft

df_ft_train = get_ft('train')
df_ft_dev = get_ft('dev')
df_ft_test = get_ft('test')


df_ft_train_dev = pd.concat([df_ft_train, df_ft_dev])

df_ft_train_dev.sample(frac=1).to_csv("/scratch/w/wluyliu/yananc/df_nerd_train.csv", index=False)
df_ft_test.sample(frac=1).to_csv("/scratch/w/wluyliu/yananc/df_nerd_test.csv", index=False)














sl = random.sample(ds_nerd['validation'], 1)[0]



for i in (ds_nerd['validation']):
    print(' '.join(i['tokens']), '\n')





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












