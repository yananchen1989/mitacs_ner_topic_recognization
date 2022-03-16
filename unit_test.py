#############  arxiv ##########
import pandas as pd 
import json
# import seaborn as sns
# import matplotlib.pyplot as plt
target_categories = ['cond-mat.supr-con', 'math.QA', 'quant-ph', 'stat.CO']

def remove_latex(sent):
    tokens = sent.replace('\n',' ').strip().split()
    tokens_clean = [ii for ii in tokens if  "$" not in ii and ii and '\\' not in ii]
    return ' '.join(tokens_clean)

from nltk.stem.porter import PorterStemmer
def clean_title(title):
    title = title.lower()
    title = re.sub(r'[^\w\s]',' ',title)
    title = title.replace('\n',' ')
    title = title.replace('_line_', ' ')
    # PorterStemmer().stem()
    tokens = [ w.strip() \
               for w in title.split(' ') if w not in sw and w and not w.isdigit() \
             and w not in string.punctuation and w not in string.ascii_lowercase and len(w) >=2 and len(w)<=15]
    if not tokens:
        return ""
    else:
        return " ".join(tokens)

def make_df(file_path):
    infos = []
    # include your local path
    with open('arxiv-metadata-oai-snapshot_2.json', 'r') as f: 
        for line in f:
            js = json.loads(line)
            if js['categories'] == 'quant-ph': # ion trap // 
                js['abstract'] = clean_text(js['abstract'])
                infos.append(js)

    df = pd.DataFrame(infos) # 78160
    #df['abstract'] = df['abstract'].map(lambda x: x.lower())
    df.drop_duplicates(['abstract'], inplace=True)
    df['yymm'] = pd.to_datetime(df['update_date'].map(lambda x: '-'.join(x.split('-')[:2] )))
    return df 

df = make_df()
print(df.sample(1)["abstract"].tolist()[0])





####################################################################################################################




authos_infos = []
anthors_set = set()
for ix, row in df.iterrows():
    i = row['authors_parsed']

    anthors = []
    for j in i:
        author = ' '.join(j).strip()
        if '\n' in author:
            author = author.split('\n')[0]
        if '  ' in author:
            author = author.split('  ')[0]
        anthors.append(author)
        authos_infos.append((author, row['title']))

    anthors_set.update(anthors)

    #print(anthors)

print(len(anthors_set))



df_ab = pd.DataFrame(authos_infos, columns=['author', 'title'])



df['author_len'] = df['authors_parsed'].map(lambda x: len(x))







import transformers

import tensorflow as tf 

MAX_LEN = 16
ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
tok = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

model = transformers.TFRobertaModel.from_pretrained('roberta-base')



model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1,x2])
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


last_hidden_states = outputs.last_hidden_state




from transformers import RobertaTokenizer, TFRobertaForSequenceClassification
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = TFRobertaForSequenceClassification.from_pretrained("roberta-base")

x = model(input_ids=ids, attention_mask=att)


model = transformers.TFRobertaForTokenClassification.from_pretrained("roberta-base")














from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df, test_size=0.1)





with open ("./finetune/df_arxiv.train.quantph.txt", 'w') as f:
    for ix, row in df_train.iterrows():
        f.write(row['abstract'] + '\n')


with open ("./finetune/df_arxiv.test.quantph.txt", 'w') as f:
    for ix, row in df_test.iterrows():
        f.write(row['abstract'] + '\n')





import random
import datasets
#raw_datasets = datasets.load_dataset('davanstrien/crowdsourced-keywords')

ds_wiki = datasets.load_dataset('wikipedia', '20200501.en')


from sentence_transformers import SentenceTransformer
#
from sklearn.metrics.pairwise import cosine_distances,cosine_similarity


model = SentenceTransformer('./finetune/arxiv_scibert', device='cuda:0')

model = SentenceTransformer('all-mpnet-base-v2', device='cuda:1', cache_folder='./cache_sentbert')

model = SentenceTransformer('allenai/scibert_scivocab_uncased', device='cuda:1', cache_folder='./cache_sentbert' )


terms = random.sample(ds_wiki['train']['title'], 1024*256)
embed_wiki = model.encode( terms, show_progress_bar=True, batch_size=1024)

terms_set = set(ds_wiki['train']['title'])


'Quantum Annealing'

embed_tag = model.encode(['Bell\'s inequalities'], show_progress_bar=True, batch_size=8)


cosine_similarity(embed_tag, embed_tag1)


embeds_score = cosine_similarity(embed_tag, embed_wiki)


dft = pd.DataFrame(zip(terms, list(embeds_score[0])), columns=['term', 'score'])


dft.sort_values(by=['score'], ascending=False, inplace=True)

for ix, row in dft.head(16).iterrows():
    print(row['term'], row['score'])


















from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased", cache_dir='./cache')
model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased", cache_dir='./cache')

sent = df.sample(1)["abstract"].tolist()[0]
tokenizer(sent)

