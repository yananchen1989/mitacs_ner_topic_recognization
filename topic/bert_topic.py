import json

# https://arxiv.org/help/api/user-manual


import pandas as pd 
from bertopic import BERTopic

from sentence_transformers import SentenceTransformer
#embedding_model = SentenceTransformer("all-mpnet-base-v2", device='cpu', cache_folder='./cache_sentbert')
embedding_model = SentenceTransformer("/Users/yanan/Downloads/finetune/arxiv_scibert_quantph", device='cpu')
embeddings = model.encode(df.sample(2048)['abstract'].tolist(), show_progress_bar=True, batch_size=64)



topic_model = BERTopic(embedding_model=embedding_model, verbose=True, min_topic_size=50)
topics, probs = topic_model.fit_transform(df['abstract'].tolist())

topic_model.save("mpnet_topic_model_sci")



topic_model = BERTopic.load("mpnet_topic_model_sci")

len(topic_model.get_topic_info())

topic_model.get_topic_info().head(10)

topic_model.get_topic(-1)



for ix, row in topic_model.get_topic_info(8).iterrows():
    print(row['Topic'], row['Count'], row['Name'])
    topic_model.get_topic(row['Topic'])
    print()




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











