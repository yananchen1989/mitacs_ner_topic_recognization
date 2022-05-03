import pandas as pd 
import argparse,random
from utils.process_func import * 


# https://maartengr.github.io/BERTopic/api/bertopic.html

parser = argparse.ArgumentParser()
parser.add_argument("--dsn", type=str)
parser.add_argument("--min_topic_size", type=int)
parser.add_argument("--num_topics", type=int)
parser.add_argument("--embedm", type=str)
args = parser.parse_args()

dfi = load_dsn(args.dsn)

from bertopic import BERTopic

######## load model 
from sentence_transformers import SentenceTransformer

if args.embedm == 'mpnet':
    embedding_model = SentenceTransformer("all-mpnet-base-v2", device='cuda', cache_folder='./cache_sentbert')
elif args.embedm == 'scibert':
    embedding_model = SentenceTransformer("allenai/scibert_scivocab_uncased", device='cuda', cache_folder='./cache_sentbert')

#for min_topic_size in [16, 32, 64, 128, 256]:

topic_model = BERTopic(embedding_model=embedding_model, verbose=True, min_topic_size=args.min_topic_size, nr_topics=args.num_topics)
topics, probs = topic_model.fit_transform(dfi['abstract_stem'].tolist())
print("min_topic_size:{}".format(args.min_topic_size) )
print("number of topics:{}".format(len(topic_model.get_topic_info())))

for i in range(len(topic_model.get_topic_info())):
    print("topic==>{}".format(i-1)) 
    for ii in topic_model.get_topic(i-1):
        try:
            print(ii[0], round(ii[1],4) )    
        except:
            continue
    print()


topic_model.save("mpnet_topic_model")

#topic_model = BERTopic.load("mpnet_topic_model")












'''


# topic-words
topic_model.visualize_barchart(top_n_topics=len(topic_model.get_topic_info()), n_words=12, height=400)


# term score decline for each topic
topic_model.visualize_term_rank()
topic_model.visualize_term_rank(log_scale=True)

# Topic Relationships
topic_model.visualize_topics(top_n_topics=len(topic_model.get_topic_info()))

topic_model.visualize_hierarchy(top_n_topics=len(topic_model.get_topic_info()), width=800)

# Topics over Time
topics_over_time = topic_model.topics_over_time(df['abstract'].tolist(), topics, df['update_date'].tolist())
topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=len(topic_model.get_topic_info()), \
                width=900, height=500, normalize_frequency=True)

'''

'''

################################# implement from scratch ################################# 

embeddings = embedding_model.encode(dfi['abstract_stem'].tolist(), show_progress_bar=True, batch_size=512)


import umap,hdbscan
umap_embeddings = umap.UMAP(n_neighbors=15, 
                            n_components=5, 
                            metric='cosine').fit_transform(embeddings)

cluster = hdbscan.HDBSCAN(min_cluster_size=5,
                          metric='euclidean',                      
                          cluster_selection_method='eom').fit(umap_embeddings)

dfs['label'] = cluster.labels_

print("pred label:\n", dfs.label.value_counts())  


# result = pd.DataFrame(umap_data, columns=['x', 'y'])
# result['label'] = cluster.labels_

# # Visualize clusters
# fig, ax = plt.subplots(figsize=(20, 10))
# outliers = result.loc[result.labels == -1, :]
# clustered = result.loc[result.labels != -1, :]
# plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=0.05)
# plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=0.05, cmap='hsv_r')
# plt.colorbar()

# docs_df = pd.DataFrame(data, columns=["Doc"])
# docs_df['Topic'] = cluster.labels_
# docs_df['Doc_ID'] = range(len(docs_df))
# docs_per_topic = docs_df.groupby(['Topic'], as_index = False).agg({'Doc': ' '.join})



docs_per_topic = dfs.groupby(['label'], as_index = False).agg({'abstract': ' '.join})



import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def c_tf_idf(documents, m, ngram_range=(1, 1)):
    count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)

    return tf_idf, count
  
tf_idf, count = c_tf_idf(docs_per_topic['abstract'].values, m=dfs.shape[0], ngram_range=(1,3))

def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
    words = count.get_feature_names()
    labels = list(docs_per_topic['label'])
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
    return top_n_words

def extract_topic_sizes(df):
    topic_sizes = (df.groupby(['label'])
                     .abstract
                     .count()
                     .reset_index()
                     .rename({"label": "label", "abstract": "Size"}, axis='columns')
                     .sort_values("Size", ascending=False))
    return topic_sizes

top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20)
topic_sizes = extract_topic_sizes(dfs)

print("topic_sizes==>\n", topic_sizes)

for l, kws in top_n_words.items():
    print(l, ' '.join([j[0] for j in kws]))


'''


