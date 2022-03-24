import datasets,argparse
from utils.process_func import * 

df = make_df('/home/w/wluyliu/yananc/nlp4quantumpapers/arxiv-metadata-oai-snapshot.json', ['quant-ph'])

print("writing")
with open('arxiv_abstract_stem', 'w') as f:
    for ix, row in df.iterrows():
        f.write(row['abstract_stem'].lower()+'\n')



########################### classification

def get_cate(cate):
    if ' ' in cate:
        tokens = cate.split()
        return tokens[-1]
    else:
        return cate

infos = []
with open('/home/w/wluyliu/yananc/nlp4quantumpapers/arxiv-metadata-oai-snapshot.json', 'r') as f: 
    for line in f:
        js = json.loads(line)
        infos.append(js)
df = pd.DataFrame(infos) # 78160
df.drop_duplicates(['abstract'], inplace=True)
df['yymm'] = pd.to_datetime(df['update_date'].map(lambda x: '-'.join(x.split('-')[:2] )))
df['abstract_clean'] = df['abstract'].map(lambda x: remove_latex(x))

df['cate'] = df['categories'].map(lambda x: get_cate(x).lower().replace('.','-'))



from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df, test_size=0.05)

with open('abstract_cate.train', 'w') as f:
    for ix, row in df_train.iterrows():
        f.write('__label__{} {}\n'.format(row['cate'], row['abstract_clean']))

with open('abstract_cate.test', 'w') as f:
    for ix, row in df_test.iterrows():
        f.write('__label__{} {}\n'.format(row['cate'], row['abstract_clean']))





import fasttext

seed_words = ['Neutral atoms', 'Quantum Annealing', 'Blockchain', 'Microwave', \
                'Casimir-type device', 'superconducting','Trapped Ion', 'Silicon']

models['skipgram']  = {}
models['cbow']  = {}

for m in models.keys():
    for ngram in [1,3]:
        models[m][ngram] = fasttext.load_model("./arxiv_abstract_embed_{}_{}.bin".format(m, ngram))

vocab = models['cbow'][1].words
print(len(vocab))


for m in models.keys():
    for ngram in [1,3]:
        print(m, ngram)
        for word in seed_words:
            print(word.lower(), '===>')
            print(models[m][ngram].get_nearest_neighbors(word.lower()))
        print()



















