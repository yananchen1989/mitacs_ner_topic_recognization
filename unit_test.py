#############  arxiv ##########
import pandas as pd 
import json


df_1 = pd.read_csv("./utils/Quantum_computing_companies.csv", header=None)
df_2 = pd.read_csv("./utils/Investors_in_quantum_computing.csv", header=None)

companies = list(set(df_1[0].tolist() + df_2[0].tolist()))


def check_entity(content):
    fallin = []
    for e in companies:
        if e.lower() in content.lower():
            fallin.append(e.lower())
    return fallin


with open('./articles_full.json', 'r') as f:
    jxml = json.load(f)



fallins = []
for js in jxml:
    content = js['post_content'] 
    fallin = check_entity(content)
    break 


sum([1 if f == 0 else 0 for f in fallins ]) / len(fallins)





####################################################################################################################








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

