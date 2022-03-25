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


dfi = load_dsn(args.dsn)

common_dictionary = Dictionary([i.split() for i in dfi['abstract_stem'].tolist()  ], prune_at=10000)

print (len(common_dictionary.token2id) )
#common_dictionary.filter_extremes(no_below=5, no_above=0.95, keep_n=10000)
#common_dictionary.save("common_dictionary")
#common_dictionary = Dictionary.load("common_dictionary")

dfi['corpus'] = dfi['abstract_stem'].map(lambda x: common_dictionary.doc2bow(x.split(' ')))

#common_dictionary.token2id['computer']

perplexity_logger = PerplexityMetric(corpus=dfi['corpus'].tolist(), logger='shell')


lda = ldamodel.LdaModel(dfi['corpus'].tolist(), id2word=common_dictionary, \
          num_topics=args.num_topics, iterations=100 , passes=10,  eval_every=10, callbacks=[perplexity_logger])

#temp_file = datapath("/scratch/w/wluyliu/yananc/lda_{}_{}".format(args.dsn, args.num_topics))
#lda.save(temp_file)


#temp_file = datapath("/gpfs/fs0/scratch/w/wluyliu/yananc/lda_arxiv_128" )
#lda = ldamodel.LdaModel.load(temp_file)

for t in range(args.num_topics):
    print("topic==>{}".format(t))
    kws = lda.show_topic(t, topn=10)
    for ii in kws:
        print(ii[0], round(ii[1],4) )
    print()


accs = []

while True:
    ls1 = random.sample(list(dfi['label'].unique()), 1)[0]
    ls2 = random.sample(list(dfi['label'].unique()), 1)[0]

    sent1 = dfi.loc[dfi['label']==ls1].sample(1)['abstract_stem'].tolist()[0]
    sent2 = dfi.loc[dfi['label']==ls2].sample(1)['abstract_stem'].tolist()[0]

    if sent1 == sent2:
        continue

    other_corpus1 = [common_dictionary.doc2bow(sent1.split(' '))]
    pred_topic1 = lda.inference(other_corpus1)[0][0].argmax()

    other_corpus2 = [common_dictionary.doc2bow(sent2.split(' '))]
    pred_topic2 = lda.inference(other_corpus2)[0][0].argmax()

    if (ls1 == ls2 and pred_topic1==pred_topic2) or (ls1!=ls2 and pred_topic1!=pred_topic2):
        accs.append(1)
    else:
        accs.append(0)

    if len(accs) % 100 == 0:
        print(sum(accs) / len(accs))



#vector = lda[other_corpus[1]]  # get topic probability distribution for a document

# Update the model by incrementally training on the new corpus.
#lda.update(other_corpus)
'''



preds = lda.inference(df.sample(10)['abstract'].tolist())






lda.update(other_corpus)
lda_v1.log_perplexity(other_corpus)


'''























