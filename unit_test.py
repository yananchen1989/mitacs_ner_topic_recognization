


#############  arxiv ##########
import pandas as pd 
import json
# import seaborn as sns
# import matplotlib.pyplot as plt
target_categories = ['cond-mat.supr-con', 'math.QA', 'quant-ph', 'stat.CO']

def clean_text(sent):
    tokens = sent.replace('\n',' ').strip().split()
    tokens_clean = [ii for ii in tokens if not ii.startswith("$") and not ii.endswith("$") and ii]
    return ' '.join(tokens_clean)

def make_df():
    infos = []
    with open('arxiv-metadata-oai-snapshot_2.json', 'r') as f: # /Users/yanan/Downloads/
        for line in f:
            js = json.loads(line)
            if js['categories'] == 'quant-ph' :
                js['abstract'] = clean_text(js['abstract'])
                infos.append(js)

    df = pd.DataFrame(infos) # 78160
    #df['abstract'] = df['abstract'].map(lambda x: x.lower())
    df.drop_duplicates(['abstract'], inplace=True)
    df['yymm'] = pd.to_datetime(df['update_date'].map(lambda x: '-'.join(x.split('-')[:2] )))
    return df 

df = make_df()
print(df.sample(1)["abstract"].tolist()[0])

from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df, test_size=0.1)


with open ("./finetune/df_arxiv.train.quantph.txt", 'w') as f:
    for ix, row in df_train.iterrows():
        f.write(row['abstract'] + '\n')


with open ("./finetune/df_arxiv.test.quantph.txt", 'w') as f:
    for ix, row in df_test.iterrows():
        f.write(row['abstract'] + '\n')







import datasets
raw_datasets = datasets.load_dataset('davanstrien/crowdsourced-keywords')





from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased", cache_dir='./cache')
model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased", cache_dir='./cache')

sent = df.sample(1)["abstract"].tolist()[0]
tokenizer(sent)

