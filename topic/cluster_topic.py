
import pandas as pd 

dfs = df.sample(2048)
embeddings = model.encode(dfs['abstract'].tolist(), show_progress_bar=True, batch_size=512)


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











