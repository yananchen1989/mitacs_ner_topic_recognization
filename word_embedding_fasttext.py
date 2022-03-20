import datasets,argparse
from utils.process_func import * 

df = make_df('/home/w/wluyliu/yananc/nlp4quantumpapers/arxiv-metadata-oai-snapshot.json', ['quant-ph'])


with open('arxiv_abstract', 'w') as f:
    for ix, row in df.iterrows():
        f.write(row['abstract_clean']+'\n')


import fasttext

for m in ['skipgram', 'cbow']
    for ngram in [1, 3]:
        model = fasttext.train_unsupervised(input='/home/w/wluyliu/yananc/nlp4quantumpapers/arxiv_abstract', 
            lr=0.1, epoch=12, wordNgrams=ngram, thread=48, dim=128, minn=0, \
            maxn=0, loss='softmax', minCount=3, model=m )
        model.save_model("/scratch/w/wluyliu/yananc/arxiv_abstract_embed_{}_{}.bin".format(m, ngram))







########################### classification
import json
import pandas as pd 
infos = []
with open('./arxiv-metadata-oai-snapshot.json', 'r') as f: 
    for line in f:
        js = json.loads(line)
        infos.append(js)
df = pd.DataFrame(infos) 


def get_cate(cate):
    if ' ' in cate:
        tokens = cate.split()
        return tokens[-1]
    else:
        return cate

df['cate'] = df['categories'].map(lambda x: get_cate(x).lower().replace('.','-'))



from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df, test_size=0.05)

with open('abstract_train_cate', 'w') as f:
    for ix, row in df_train.iterrows():
        f.write('__label__{} {}\n'.format(row['cate'], row['abstract_clean']))

with open('abstract_test_cate', 'w') as f:
    for ix, row in df_test.iterrows():
        f.write('__label__{} {}\n'.format(row['cate'], row['abstract_clean']))


for ep in [7, 12, 25, 30, 50, 70, 100]:
    model = fasttext.train_supervised(input="./abstract_train_cate", dim=128,\
             loss='softmax', thread=56,  minn=1, maxn=5, wordNgrams=4, epoch=ep)
    
    result = model.test("abstract_test_cate")
    print("epoch:{}".format(ep), result)
    model.save_model("./abstract_cate_ep{}.bin".format(ep))





model = fasttext.load_model("./abstract_cate.bin")

vocab = model.words
model.get_word_vector("the")
model.get_nearest_neighbors('asparagus')



'''
show unitar allow clone two point ray implic clone geometr phase inform quantum state particular quantum histori encod geometr phase cyclic evolut quantum system cannot copi also prove gener geometr phase inform cannot copi unitari oper argu result also hold consist histori formul quantum mechan

appli quantum control techniqu control larg spin chain act two qubit one end therebi implement univers quantum comput combin quantum gate latter swap oper across chain shown control sequenc comput implement effici discuss applic idea physic system superconduct qubit full control long chain challeng

propos reliabl scheme simul tunabl ultrastrong mix first order quadrat optomechan coupl coexist optomechan interact coupl two mode boson system two mode coupl cross kerr interact one two mode driven singl two excit process show mix optomechan interact enter singl photon strong coupl even ultrastrong coupl regim strength first order quadrat optomechan coupl control demand henc first order quadrat mix optomechan model realiz particular thermal nois driven mode suppress total introduc proper squeez vacuum bath also studi gener superposit coher squeez state vacuum state base simul interact quantum coher effect gener state character calcul wigner function close open system case work pave way observ applic ultrastrong optomechan effect quantum simul

quantum error correct necessari perform larg scale quantum comput presenc nois decoher result sever aspect quantum error correct alreadi explor primarili studi quantum memori import first step toward quantum comput object increas lifetim encod quantum inform addit sever work explor implement logic gate work studi next step fault tolerantli implement quantum circuit choos bacon shor subsystem code particularli simpl error detect circuit numer site count argument comput pseudo threshold pauli error rate depolar nois model encod circuit outperform unencod circuit pseudo threshold valu shown high short circuit circuit moder depth addit see multipl round stabil measur give improv perform singl round end provid concret suggest small scale fault toler demonstr quantum algorithm could access exist hardwar

zx calculu mathemat tool repres analys quantum oper manipul diagram effect repres tensor network two famili node network one commut either rotat rotat usual call green node red node respect origin formul zx calculu motiv part properti algebra form green red node notabl form bialgebra scalar factor consequ diagram transform notat certain unitari oper involv scalar gadget denot contribut normalis factor present renormalis gener zx calculu form bialgebra precis result scalar gadget requir repres common unitari transform correspond diagram transform gener simpler also present similar renormalis version zh calculu obtain result analysi condit variou idealis rewrit sound leverag exist present zx zh calculi

formul wave atom optic theori collect atom recoil laser atom center mass motion treat quantum mechan compar predict theori ray atom optic theori treat center mass motion classic show case far reson pump laser ray optic model fail predict linear respons carl temperatur order recoil temperatur less due fact thei temperatur regim one longer ignor effect matter wave diffract atom center mass motion


__label__sauce __label__cheese How much does potato starch affect a cheese sauce recipe?
__label__food-safety __label__acidity Dangerous pathogens capable of growing in acidic environments
__label__cast-iron __label__stove How do I cover up the white spots on my cast iron stove?
__label__restaurant Michelin Three Star Restaurant; but if the chef is not there
__label__knife-skills __label__dicing Without knife skills, how can I quickly and accurately dice vegetables?
__label__storage-method __label__equipment __label__bread What's the purpose of a bread box?
__label__baking __label__food-safety __label__substitutions __label__peanuts how to seperate peanut oil from roasted peanuts at home?
__label__chocolate American equivalent for British chocolate terms
__label__baking __label__oven __label__convection Fan bake vs bake
'''