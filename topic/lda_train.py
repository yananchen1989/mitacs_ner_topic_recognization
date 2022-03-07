
import pandas as pd
import datetime,random,hashlib,traceback,json,os,logging,time,pickle,gc,requests,operator,argparse
import numpy as np
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.test.utils import datapath
from gensim.models import ldamodel

import re, string
from nltk.corpus import stopwords
sw = set(stopwords.words("english"))
# from nltk.stem.porter import PorterStemmer


df['abstract'] = df['abstract'].map(lambda x: x.lower())
common_dictionary = Dictionary([i.split() for i in df['abstract'].tolist()  ], prune_at=10000)

print (len(common_dictionary.token2id) )
#common_dictionary.filter_extremes(no_below=5, no_above=0.95, keep_n=10000)
common_dictionary.save("common_dictionary")

common_dictionary = Dictionary.load("common_dictionary")

from gensim.models import ldamodel

####################  train


df['corpus'] = df['abstract'].map(lambda x: common_dictionary.doc2bow(x.split(' ')))

#common_dictionary.token2id['computer']

from gensim.models.callbacks import PerplexityMetric
from gensim.models.callbacks import CallbackAny2Vec


perplexity_logger = PerplexityMetric(corpus=df['corpus'].tolist(), logger='shell')




lda = ldamodel.LdaModel(df['corpus'].tolist(), id2word=common_dictionary, \
          num_topics=64, iterations=100 , distributed=True, eval_every=10, callbacks=[perplexity_logger])

temp_file = datapath("./lda_abstract" )
lda.save(temp_file)


lda.show_topics(num_topics=64, num_words=10)

temp_file = datapath("./lda_abstract" )

lda = ldamodel.LdaModel.load(temp_file)

for t in range(64):
    kws = lda.show_topic(t, topn=20)
    for ii in kws:
        if ii[0].lower() in sw:
            continue
        else:
            print(ii)
    print()



#vector = lda[other_corpus[1]]  # get topic probability distribution for a document

# Update the model by incrementally training on the new corpus.
#lda.update(other_corpus)


other_corpus = [common_dictionary.doc2bow(sent.split(' '))]

preds = lda.inference(df.sample(10)['abstract'].tolist())

lda.inference(other_corpus)


dfs['topic'] = preds[0].argmax(axis=1) 



dfs.loc[dfs['topic']==604]['content']



lda.update(other_corpus)
lda_v1.log_perplexity(other_corpus)


























