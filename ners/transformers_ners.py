import json,random
import pandas as pd 

# huggingface transformers
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

# conll2003
# tokenizer_bert = AutoTokenizer.from_pretrained("dslim/bert-base-NER", cache_dir='/scratch/w/wluyliu/yananc/cache')
# model_bert = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER", cache_dir='/scratch/w/wluyliu/yananc/cache')

# conll2003
tokenizer_roberta = AutoTokenizer.from_pretrained("Jean-Baptiste/roberta-large-ner-english", cache_dir='/scratch/w/wluyliu/yananc/cache')
model_roberta = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/roberta-large-ner-english", cache_dir='/scratch/w/wluyliu/yananc/cache')

# nlp_bert = pipeline("ner", model=model_bert, tokenizer=tokenizer_bert, aggregation_strategy="simple")
nlp_roberta = pipeline("ner", model=model_roberta, tokenizer=tokenizer_roberta, aggregation_strategy="simple")


path_local = "/scratch/w/wluyliu/yananc/finetunes/roberta_nerd_fine" 
tokenizer_roberta = AutoTokenizer.from_pretrained(path_local)
model_roberta = AutoModelForTokenClassification.from_pretrained(path_local)



def check_entity_ft(sent):
    ner_results = nlp_roberta(sent)
    if not ner_results:
        return []
    infos = []
    for ii in ner_results:
        if ii['word'].strip() in string.punctuation:
            continue
        infos.append((ii['entity_group'], ii['word'].strip()))
    return infos


# spacy
# https://spacy.io/models/en#en_core_web_lg
# python -m spacy download en_core_web_lg
# python -m spacy download en_core_web_trf
import spacy
#ner_spacy_model = spacy.load('en_core_web_lg', disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
nlp_split_sents = spacy.load('en_core_web_sm')


# flair 
from flair.data import Sentence
from flair.models import SequenceTagger
import torch, flair, string
#flair.device = torch.device('cpu')
flair.device = torch.device('cuda:0')
#tagger = SequenceTagger.load("flair/ner-english-fast")
# https://github.com/flairNLP/flair/blob/master/flair/data.py
tagger_conll = SequenceTagger.load("flair/ner-english-large")
tagger_ontonotes = SequenceTagger.load("flair/ner-english-ontonotes-large")

def check_entity_flair(content, tagger):
    sentence = Sentence(content)
    tagger.predict(sentence)
    # ii.start_position, ii.end_position
    ners = sentence.get_spans('ner')
    infos = []
    for ii in sentence.get_spans('ner'):
        if (ii.tag, ii.text) in infos:
            continue
        if ii.text.strip() in string.punctuation:
            continue
        infos.append((ii.tag, ii.text.strip()))
    return infos




####### nli ###############
# nli_nlp_bart = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0) #  1.8.1+cu102
# vicgalle/xlm-roberta-large-xnli-anli joeddav/xlm-roberta-large-xnli 
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model_nli = AutoModelForSequenceClassification.from_pretrained('vicgalle/xlm-roberta-large-xnli-anli', cache_dir='/scratch/w/wluyliu/yananc/cache')
tokenizer_nli = AutoTokenizer.from_pretrained('vicgalle/xlm-roberta-large-xnli-anli', cache_dir='/scratch/w/wluyliu/yananc/cache')
nli_nlp_roberta = pipeline("zero-shot-classification", model=model_nli, tokenizer=tokenizer_nli, device=0)


############## main ########################


with open('/home/w/wluyliu/yananc/nlp4quantumpapers/articles_full.json', 'r') as f:
    jxml = json.load(f)

random.shuffle(jxml)
content = random.sample(jxml, 1)[0]['post_content']



sents = [i.text for i in nlp_split_sents(content).sents]

infos = []
for sent in sents:
    flair_ontonotes_ner_results = check_entity_flair(sent, tagger_ontonotes)
    flair_conll_ner_results = check_entity_flair(sent, tagger_conll)
    ft_ner_results = check_entity_ft(sent)
    infos.extend(flair_ontonotes_ner_results)
    infos.extend(flair_conll_ner_results)
    infos.extend(ft_ner_results)
    print(sent) 

df = pd.DataFrame(infos, columns=['ner','span'])
df['ner'] = df['ner'].map(lambda x: 'PER' if x=='PERSON' else x )
df.drop_duplicates(['ner','span'], inplace=True)
df.sort_values(['ner', 'span'], inplace=True)

for ix, row in df.iterrows():
    print(row['ner'], '\t', row['span'])


labels_candidates = ['company', 'investor', 'university','Venture Capital Investor', 'academic organization', 
                        'goverment organization']

for span in df.loc[df['ner']=='ORG']['span'].tolist():
    result_nli = nli_nlp_roberta(content, labels_candidates, multi_label=True, hypothesis_template= span + " is a {}.")
    dfr = pd.DataFrame(zip(result_nli['labels'], result_nli['scores']) , columns=['cate','score'])
    print(span, '\n', dfr)

















