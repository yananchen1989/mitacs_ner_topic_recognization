#############  arxiv ##########
import pandas as pd 
import json,random
from flair.data import Sentence
from flair.models import SequenceTagger

import torch, flair
#flair.device = torch.device('cpu')
flair.device = torch.device('cuda:0')
#  note that get_ners can only run one single GPU !!! 
#tagger = SequenceTagger.load("flair/ner-english-fast")
# https://github.com/flairNLP/flair/blob/master/flair/data.py
tagger = SequenceTagger.load("flair/ner-english-large")

df_1 = pd.read_csv("./datasets/Quantum_computing_companies.csv", header=None)
df_2 = pd.read_csv("./datasets/Investors_in_quantum_computing.csv", header=None)

companies = list(set(df_1[0].tolist() + df_2[0].tolist()))


def check_entity_predefine(content):
    fallin = []
    for e in companies:
        if e.lower() in content.lower():
            fallin.append(e)
    return fallin

def check_entity_model(content):
    sentence = Sentence(content)
    tagger.predict(sentence)

    ners = sentence.get_spans('ner')
    fallin = set()
    for ii in sentence.get_spans('ner'):
        if ii.tag == 'ORG':
            fallin.add(ii.text)
    return list(fallin)

# for ii in sentence.get_spans('ner'):
#     print(ii.text, ii.tag, ii.start_position, ii.end_position)



with open('./articles_full.json', 'r') as f:
    jxml = json.load(f)

random.shuffle(jxml)

infos = []
for js in jxml:
    content = js['post_content'].strip()
    fallin_predefine = check_entity_predefine(content)
    fallin_model = check_entity_model(content)
    infos.append((js['article_ID'], len(fallin_predefine), len(fallin_model)) )
    print("predefine==>\n", ', '.join(fallin_predefine))
    print("model==>\n", ', '.join(fallin_model))
    print('\n')







'''


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


'''

























