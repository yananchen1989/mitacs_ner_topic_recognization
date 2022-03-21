import fasttext

m = 'skipgram' # cbow
model = fasttext.train_unsupervised(input='/home/w/wluyliu/yananc/nlp4quantumpapers/arxiv_abstract_stem', 
    lr=0.1, epoch=20, wordNgrams=1, thread=36, dim=128, minn=0, \
    maxn=0, loss='softmax', minCount=3, model=m )
model.save_model("/scratch/w/wluyliu/yananc/arxiv_abstract_embed_{}.bin".format(m))
