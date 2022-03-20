import fasttext

for m in ['skipgram', 'cbow']
    for ngram in [1, 3]:
        model = fasttext.train_unsupervised(input='/home/w/wluyliu/yananc/nlp4quantumpapers/arxiv_abstract', 
            lr=0.1, epoch=12, wordNgrams=ngram, thread=48, dim=128, minn=0, \
            maxn=0, loss='softmax', minCount=3, model=m )
        model.save_model("/scratch/w/wluyliu/yananc/arxiv_abstract_embed_{}_{}.bin".format(m, ngram))
