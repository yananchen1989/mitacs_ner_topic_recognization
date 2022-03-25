import pandas as pd
import datetime,random,hashlib,traceback,json,os,logging,time,pickle,gc,requests,operator,argparse
import numpy as np
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.test.utils import datapath
from gensim.models import ldamodel
# from nltk.stem.porter import PorterStemmer
import datasets,argparse
from utils.process_func import * 

from gensim.models.callbacks import PerplexityMetric
from gensim.models.callbacks import CallbackAny2Vec


parser = argparse.ArgumentParser()
parser.add_argument("--dsn", type=str)
parser.add_argument("--num_topics", type=int)
args = parser.parse_args()

if args.dsn == 'cc':
    cc_news = datasets.load_dataset('cc_news', split="train", cache_dir='/scratch/w/wluyliu/yananc/cache')
    df = pd.DataFrame(random.sample(cc_news['text'], 100000), columns=['abstract'])
elif args.dsn == 'arxiv':
    df = make_df('/home/w/wluyliu/yananc/nlp4quantumpapers/arxiv-metadata-oai-snapshot_2.json', ['quant-ph'])

elif args.dsn == 'subgroup':
    # cate sub
    df = pd.read_csv("./artificially_labeled_abstracts.csv")
    dfi = df.loc[df['Assigned_group']=='Full stack and quantum computers']
    dfi['abstract_clean'] = dfi['abstract'].map(lambda x: remove_latex(x))
    dfi['abstract_stem'] = dfi['abstract_clean'].map(lambda x: clean_title(x))



common_dictionary = Dictionary([i.split() for i in dfi['abstract_stem'].tolist()  ], prune_at=10000)

print (len(common_dictionary.token2id) )
#common_dictionary.filter_extremes(no_below=5, no_above=0.95, keep_n=10000)
#common_dictionary.save("common_dictionary")
#common_dictionary = Dictionary.load("common_dictionary")

dfi['corpus'] = dfi['abstract_stem'].map(lambda x: common_dictionary.doc2bow(x.split(' ')))

#common_dictionary.token2id['computer']

perplexity_logger = PerplexityMetric(corpus=dfi['corpus'].tolist(), logger='shell')


lda = ldamodel.LdaModel(dfi['corpus'].tolist(), id2word=common_dictionary, \
          num_topics=10, iterations=100 , passes=10,  eval_every=10, callbacks=[perplexity_logger])

#temp_file = datapath("/scratch/w/wluyliu/yananc/lda_{}_{}".format(args.dsn, args.num_topics))
#lda.save(temp_file)


#temp_file = datapath("/gpfs/fs0/scratch/w/wluyliu/yananc/lda_arxiv_128" )
#lda = ldamodel.LdaModel.load(temp_file)




for t in range(10):
    print("topic==>{}".format(t))
    kws = lda.show_topic(t, topn=10)
    for ii in kws:
        print(ii[0], round(ii[1],4) )
    print()









#vector = lda[other_corpus[1]]  # get topic probability distribution for a document

# Update the model by incrementally training on the new corpus.
#lda.update(other_corpus)
'''

other_corpus = [common_dictionary.doc2bow(sent.split(' '))]

preds = lda.inference(df.sample(10)['abstract'].tolist())

lda.inference(other_corpus)


dfs['topic'] = preds[0].argmax(axis=1) 



dfs.loc[dfs['topic']==604]['content']



lda.update(other_corpus)
lda_v1.log_perplexity(other_corpus)


'''























