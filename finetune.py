from datasets import load_dataset

imdb = load_dataset("imdb")

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128)

tokenized_imdb = imdb.map(preprocess_function, batched=True, load_from_cache_file=True)


######################
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer, return_tensors="tf")

tf_train_dataset = tokenized_imdb["train"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "label"],
    shuffle=True,
    batch_size=16,
    collate_fn=data_collator,
)

tf_validation_dataset = tokenized_imdb["test"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "label"],
    shuffle=False,
    batch_size=16,
    collate_fn=data_collator,
)

from transformers import create_optimizer
import tensorflow as tf

batch_size = 16
num_epochs = 5
batches_per_epoch = len(tokenized_imdb["train"]) // batch_size
total_train_steps = int(batches_per_epoch * num_epochs)
optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)


from transformers import TFAutoModelForSequenceClassification
model = TFAutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)


#model.compile(optimizer=optimizer)

model.compile(tf.keras.optimizers.Adam(learning_rate=2e-5), "binary_crossentropy", metrics=["binary_accuracy"])
model.fit(x=tf_train_dataset, validation_data=tf_validation_dataset, epochs=3)


tf_unsupervise_dataset = tokenized_imdb["unsupervised"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "label"],
    shuffle=False,
    batch_size=16,
    collate_fn=data_collator,
)



from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df, test_size=0.1)





with open ("./finetune/df_arxiv.train.quantph.txt", 'w') as f:
    for ix, row in df_train.iterrows():
        f.write(row['abstract'] + '\n')


with open ("./finetune/df_arxiv.test.quantph.txt", 'w') as f:
    for ix, row in df_test.iterrows():
        f.write(row['abstract'] + '\n')























