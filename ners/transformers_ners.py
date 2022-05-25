import json,random,string
import pandas as pd 
import numpy as np
# huggingface transformers
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

import tensorflow as tf 
gpus = tf.config.experimental.list_physical_devices('GPU')
print('======>',gpus,'<=======')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
      # tf.config.experimental.set_virtual_device_configuration(gpu, \
      #      [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
  except RuntimeError as e:
    print(e)

# conll2003
# tokenizer_bert = AutoTokenizer.from_pretrained("dslim/bert-base-NER", cache_dir='/scratch/w/wluyliu/yananc/cache')
# model_bert = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER", cache_dir='/scratch/w/wluyliu/yananc/cache')

# load off-the-shelf model : roberta-large fine-tuned on conll2003
tokenizer_roberta = AutoTokenizer.from_pretrained("Jean-Baptiste/roberta-large-ner-english", cache_dir='/scratch/w/wluyliu/yananc/cache')
model_roberta = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/roberta-large-ner-english", cache_dir='/scratch/w/wluyliu/yananc/cache')
nlp_roberta = pipeline("ner", model=model_roberta, tokenizer=tokenizer_roberta, \
                    aggregation_strategy="simple", device=gpu)

# # load off-the-shelf model : roberta-large fine-tuned on nerd-fine-grained
tokenizer_roberta_nerd_fine = AutoTokenizer.from_pretrained("/scratch/w/wluyliu/yananc/finetunes_ner/roberta_nerd_fine")
model_roberta_nerd_fine = AutoModelForTokenClassification.from_pretrained("/scratch/w/wluyliu/yananc/finetunes_ner/roberta_nerd_coarse")
nlp_roberta_nerd_fine = pipeline("ner", model=model_roberta_nerd_fine, tokenizer=tokenizer_roberta_nerd_fine,\
                     aggregation_strategy="none", device=3)

# # load off-the-shelf model : roberta-large fine-tuned on nerd-coarse-grained
# tokenizer_roberta_nerd_coarse = AutoTokenizer.from_pretrained("/scratch/w/wluyliu/yananc/finetunes/roberta_nerd_coarse")
# model_roberta_nerd_coarse = AutoModelForTokenClassification.from_pretrained("/scratch/w/wluyliu/yananc/finetunes/roberta_nerd_coarse")
# nlp_roberta_nerd_coarse = pipeline("ner", model=model_roberta_nerd_coarse, tokenizer=tokenizer_roberta_nerd_coarse,\
#                     aggregation_strategy="simple", device=gpu)

ix = random.sample(range(len(raw_datasets_['test'])), 1)[0]
content = ' '.join(raw_datasets_['test'][ix]['tokens'])

for token, tag in zip(raw_datasets_['test'][ix]['tokens'], raw_datasets_['test'][ix]['tags']):
    if tag != 'O':
        print(tag, '===>', token )
print()


ner_results = nlp_roberta_nerd_fine(content)

if len(ner_results) > 0:

def decode_result(ner_results):
    df_tmp = pd.DataFrame(ner_results)
    for tag in df_tmp['entity'].unique():
        df_tmp_f = df_tmp.loc[df_tmp['entity']==tag]
        list_of_df = [d for _, d in df_tmp_f.groupby(df_tmp_f.index - np.arange(len(df_tmp_f)))]

        for df_tmp_f_ in list_of_df:
            ner = tokenizer_roberta_nerd_fine.decode(tokenizer_roberta_nerd_fine.convert_tokens_to_ids(df_tmp_f_['word'].tolist()),\
                                skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
            print(tag, '===>', ner)









def check_entity_ft(content, nlp_roberta, fmark):
    ner_results = nlp_roberta(content)

    infos = []
    for ii in ner_results:
        infos.append((ii['entity_group'], ii['word'].strip(), ii['start'], ii['end'], fmark))
    return infos



# spacy
# https://spacy.io/models/en#en_core_web_lg
# python -m spacy download en_core_web_lg
# python -m spacy download en_core_web_trf
import spacy
en_core_web_lg = spacy.load('en_core_web_lg', disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
nlp_split_sents = spacy.load('en_core_web_sm')

def check_entity_spacy(content, en_core_web_lg, fmark ):
    doc = en_core_web_lg(content)
    infos = []
    for ent in doc.ents:
        infos.append((ent.label_, ent.text, ent.start_char, ent.end_char, fmark))

    return infos


# flair 
from flair.data import Sentence
from flair.models import SequenceTagger
import torch, flair, string

if gpu >= 0:
    flair.device = torch.device('cuda:{}'.format(gpu))
else:
    flair.device = torch.device('cpu')

tagger_conll = SequenceTagger.load("flair/ner-english-large")
tagger_ontonotes = SequenceTagger.load("flair/ner-english-ontonotes-large")

def check_entity_flair(content, tagger, fmark):
    sentence = Sentence(content)
    tagger.predict(sentence)

    infos = []
    for ii in sentence.get_spans('ner'):
        infos.append((ii.tag, ii.text.strip(), ii.start_position, ii.end_position, fmark))
 
    return infos






############## main ########################
import spacy,re
from spacy import displacy

def find_index_of_entities(span, raw_text):
    indexs = []
    for match in re.finditer(span, raw_text):
        indexs.append((match.start(), match.end()))
    return indexs



nlp = spacy.blank('en')

def check_overlap(index_exist, sel_index):
    if not index_exist:
        return 0 

    for ii in index_exist:
        if (sel_index[0]>=ii[0] and sel_index[0]<=ii[1]) or (sel_index[1]>=ii[0] and sel_index[1]<=ii[1]):
            return 1 
    return 0




with open('/home/w/wluyliu/yananc/nlp4quantumpapers/articles_full.json', 'r') as f:
    jxml = json.load(f)

tag_map = {'PERSON':'PER', 'ORGANIZATION':'ORG', 'LOCATION':'LOC', 'GPE':'LOC'}
pick_from_nerd = ['EDUCATION','COMPANY','SOFTWARE','Government'.upper() ]


# sents = [i.text.replace('\n','').strip() for i in nlp_split_sents(content).sents]

def ner_combine(content):
    infos = []

    flair_ontonotes_ner_results = check_entity_flair(content, tagger_ontonotes,'flaironto')
    infos.extend(flair_ontonotes_ner_results)

    flair_conll_ner_results = check_entity_flair(content, tagger_conll,'flairconll')
    infos.extend(flair_conll_ner_results)

    ft_ner_ori = check_entity_ft(content, nlp_roberta, 'ftori')
    infos.extend(ft_ner_ori)


    # spacy_ner_results = check_entity_spacy(content, en_core_web_lg, 'spacy')
    # infos.extend(spacy_ner_results)

    # ft_ner_nerd_coarse = check_entity_ft(sent, nlp_roberta_nerd_coarse) 
    # infos.extend(ft_ner_nerd_coarse)

    # ft_ner_nerd_fine = check_entity_ft(sent, nlp_roberta_nerd_fine) 
    # infos.extend(ft_ner_nerd_fine)

        # print(sent) 

    df = pd.DataFrame(infos, columns=['ner','span', 'start', 'end', 'fmark'])
    df['ner'] = df['ner'].map(lambda x: x.upper())
    df['ner'] = df['ner'].map(lambda x: tag_map[x] if x in tag_map else x )

    df = df.loc[~df['span'].isin([i for i in string.punctuation])]
    df.drop_duplicates(inplace=True)

    df_org = df.loc[df['ner']=='ORG']
    return df_org
# df_pretrain = df.loc[df['fmark'].isin(['flair_onto', 'flair_conll', 'ft_ori'])]
# df_nerd = df.loc[df['fmark'].isin(['nerd_coarse', 'nerd_fine'])]
# df_nerd = df_nerd.loc[df_nerd['ner'].isin(pick_from_nerd)]

# df_pretrain.drop_duplicates(['ner','span'], inplace=True)
# df_nerd.drop_duplicates(['ner','span'], inplace=True)

# df_pretrain.sort_values(by=['ner'], ascending=True, inplace=True)
# df_nerd.sort_values(by=['ner'], ascending=True, inplace=True)
# return df_pretrain, df_nerd

samples = random.sample(jxml, 10)

import tensorflow_hub as hub
import tensorflow_text as text
model = tf.keras.models.load_model("/scratch/w/wluyliu/yananc/cls_dict")
df_tags = pd.read_csv("/home/w/wluyliu/yananc/nlp4quantumpapers/datasets/QI-NERs.csv")

ixl = {i:j for i,j in enumerate(df_tags['tag'].drop_duplicates().tolist()) }
ixl_rev = {j:i for i,j in enumerate(df_tags['tag'].drop_duplicates().tolist()) }

DEFAULT_LABEL_COLORS = {
    "ORG": "#7aecec",
    "PRODUCT": "#bfeeb7",
    "GPE": "#feca74",
    "LOC": "#ff9561",
    "PERSON": "#aa9cfc",
    "NORP": "#c887fb",
    "FAC": "#9cc9cc",
    "EVENT": "#ffeb80",
    "LAW": "#ff8197",
    "LANGUAGE": "#ff8197",
    "WORK_OF_ART": "#f0d0ff",
    "DATE": "#bfe1d9",
    "TIME": "#bfe1d9",
    "MONEY": "#e4e7d2",
    "QUANTITY": "#e4e7d2",
    "ORDINAL": "#e4e7d2",
    "CARDINAL": "#e4e7d2",
    "PERCENT": "#e4e7d2",
}



# colours = list(DEFAULT_LABEL_COLORS.values())
colours = [DEFAULT_LABEL_COLORS['GPE'], DEFAULT_LABEL_COLORS['ORG'], DEFAULT_LABEL_COLORS['PERSON'],
           DEFAULT_LABEL_COLORS['CARDINAL'], DEFAULT_LABEL_COLORS['DATE'], DEFAULT_LABEL_COLORS['WORK_OF_ART']]


random.shuffle(jxml)
# df_ll = []
with open('ner_cls_samples.jsonl', 'w') as f:
    for sample in jxml:
        content = sample['post_content'].encode("ascii", "ignore").decode("utf-8").strip()

        df_org = ner_combine(content)

        if df_org.shape[0] <=3:
            print("unnormal df_org==>", df_org.shape[0])
            continue

        preds = model.predict(df_org['span'].values.reshape(-1,1))
        preds_labels = preds.argmax(axis=1)

        preds_scores = preds.max(axis=1)

        df_org['pred_tag'] = [ixl[i] for i in preds_labels]
        df_org['pred_score'] = preds_scores
        # df_ll.append(df_org)

         
        json_tmp = {'text':content}
        spans = []
        # visualization
        doc = nlp.make_doc(content)
        ents = []
        index_exist = []

        for ix, row in df_org.iterrows():
            if len(row['span'])<=1:
                continue
            overlap = check_overlap(index_exist, (row['start'], row['end']))

            if overlap:
                continue

            ent = doc.char_span(row['start'], row['end'], label=row['pred_tag'])
            # print(row['span'], '==>', row['start'], row['end'])
            if ent is None:
                continue
            ents.append(ent)
            index_exist.append((row['start'], row['end']))
            spans.append({'start':row['start'], 'end':row['end'], 'label':row['pred_tag']})

        json_tmp['spans'] = spans


        json.dump(json_tmp, f)
        f.write('\n')








    doc.ents = ents
    options = { "ents":df_tags['tag'].drop_duplicates().tolist(),
            "colors": {ii[0]:ii[1] for ii in zip(df_tags['tag'].unique().tolist(), colours[:df_tags['tag'].unique().shape[0]])}
        }
    html = displacy.render(doc, style="ent", page=True, options=options)

    with open("ner_spacy_test100.html", "a") as ff:
        ff.write('article_ID:{}'.format(sample['article_ID'])+'\n'+ html+'------------\n\n') 

    print(sample['article_ID'])






annotations = []
with open("5_samples_annotations.jsonl", "r") as file:
    for line in file:
        annotations.append(json.loads(line))





