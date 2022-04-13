#############  ########## CUDA_VISIBLE_DEVICES
import pandas as pd 
import json,random



df_1 = pd.read_csv("./datasets/Quantum_computing_companies.csv", header=None)
df_2 = pd.read_csv("./datasets/Investors_in_quantum_computing.csv", header=None)

companies = list(set(df_1[0].tolist() + df_2[0].tolist()))


def check_entity_predefine(content):
    fallin = []
    for e in companies:
        if e.lower() in content.lower():
            fallin.append(e)
    return fallin





import datasets
ds_nerd = datasets.load_dataset('dfki-nlp/few-nerd', "supervised", cache_dir='/scratch/w/wluyliu/yananc/cache')
ds_notes = datasets.load_dataset('conll2012_ontonotesv5', "english_v12", cache_dir='/scratch/w/wluyliu/yananc/cache')
ds_conll = datasets.load_dataset('conll2003', cache_dir='/scratch/w/wluyliu/yananc/cache')

raw_datasets = ds_nerd

ds_nerd['train']['ner_tags'][35]



'''
{
    'id': '1', 
    'tokens': ['It', 'starred', 'Hicks', "'s", 'wife', ',', 'Ellaline', 'Terriss', 'and', 'Edmund', 'Payne', '.'], 
    'ner_tags': [0, 0, 7, 0, 0, 0, 7, 7, 0, 7, 7, 0], 
    'fine_ner_tags': [0, 0, 51, 0, 0, 0, 50, 50, 0, 50, 50, 0]
}
'''

sentence = data['sentence'][index].strip().split() 

max_len = 64
encoding = tokenizer(sentence,
                             #is_pretokenized=True, 
                             is_split_into_words=True,
                             return_offsets_mapping=True, 
                             padding='max_length', 
                             truncation=True, 
                             max_length=max_len)


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

























