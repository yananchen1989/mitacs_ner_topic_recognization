# -*- coding: utf-8 -*-
"""Custom Named Entity Recognition with BERT.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Custom_Named_Entity_Recognition_with_BERT_only_first_wordpiece.ipynb

## **Fine-tuning BERT for named-entity recognition**

In this notebook, we are going to use **BertForTokenClassification** which is included in the [Transformers library](https://github.com/huggingface/transformers) by HuggingFace. This model has BERT as its base architecture, with a token classification head on top, allowing it to make predictions at the token level, rather than the sequence level. Named entity recognition is typically treated as a token classification problem, so that's what we are going to use it for.

This tutorial uses the idea of **transfer learning**, i.e. first pretraining a large neural network in an unsupervised way, and then fine-tuning that neural network on a task of interest. In this case, BERT is a neural network pretrained on 2 tasks: masked language modeling and next sentence prediction. Now, we are going to fine-tune this network on a NER dataset. Fine-tuning is supervised learning, so this means we will need a labeled dataset.

If you want to know more about BERT, I suggest the following resources:
* the original [paper](https://arxiv.org/abs/1810.04805)
* Jay Allamar's [blog post](http://jalammar.github.io/illustrated-bert/) as well as his [tutorial](http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/)
* Chris Mccormick's [Youtube channel](https://www.youtube.com/channel/UCoRX98PLOsaN8PtekB9kWrw)
* Abbishek Kumar Mishra's [Youtube channel](https://www.youtube.com/user/abhisheksvnit)

The following notebook largely follows the same structure as the tutorials by Abhishek Kumar Mishra. For his tutorials on the Transformers library, see his [Github repository](https://github.com/abhimishra91/transformers-tutorials).

NOTE: this notebook assumes basic knowledge about deep learning, BERT, and native PyTorch. If you want to learn more Python, deep learning and PyTorch, I highly recommend cs231n by Stanford University and the FastAI course by Jeremy Howard et al. Both are freely available on the web.  

Now, let's move on to the real stuff!

#### **Importing Python Libraries and preparing the environment**

This notebook assumes that you have the following libraries installed:
* pandas
* numpy
* sklearn
* pytorch
* transformers
* seqeval

As we are running this in Google Colab, the only libraries we need to additionally install are transformers and seqeval (GPU version):
"""
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import torch, argparse, random
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertConfig, BertForTokenClassification
from sklearn.metrics import * 
"""As deep learning can be accellerated a lot using a GPU instead of a CPU, make sure you can run this notebook in a GPU runtime (which Google Colab provides for free! - check "Runtime" - "Change runtime type" - and set the hardware accelerator to "GPU").

We can set the default device to GPU using the following code (if it prints "cuda", it means the GPU has been recognized):
"""
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default="3", type=str)
args = parser.parse_args()

from torch import cuda
device = 'cuda:{}'.format(args.gpu) if cuda.is_available() else 'cpu'
print(device)

"""#### **Downloading and preprocessing the data**
Named entity recognition (NER) uses a specific annotation scheme, which is defined (at least for European languages) at the *word* level. An annotation scheme that is widely used is called **[IOB-tagging](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)**, which stands for Inside-Outside-Beginning. Each tag indicates whether the corresponding word is *inside*, *outside* or at the *beginning* of a specific named entity. The reason this is used is because named entities usually comprise more than 1 word. 

Let's have a look at an example. If you have a sentence like "Barack Obama was born in Hawaï", 
then the corresponding tags would be   [B-PERS, I-PERS, O, O, O, B-GEO]. 
B-PERS means that the word "Barack" is the beginning of a person, 
I-PERS means that the word "Obama" is inside a person, 
"O" means that the word "was" is outside a named entity, and so on. 
So one typically has as many tags as there are words in a sentence.

So if you want to train a deep learning model for NER, 
it requires that you have your data in this IOB format 
(or similar formats such as [BILOU](https://stackoverflow.com/questions/17116446/what-do-the-bilou-tags-mean-in-named-entity-recognition)). There exist many annotation tools which let you create these kind of annotations automatically (such as Spacy's [Prodigy](https://prodi.gy/), 
[Tagtog](https://docs.tagtog.net/) or [Doccano](https://github.com/doccano/doccano)). You can also use Spacy's [biluo_tags_from_offsets](https://spacy.io/api/goldparse#biluo_tags_from_offsets) function to convert annotations at the character level to IOB format.

Here, we will use a NER dataset from [Kaggle](https://www.kaggle.com/namanj27/ner-dataset) 
that is already in IOB format. One has to go to this web page, download the dataset, 
unzip it, and upload the csv file to this notebook. Let's print out the first few rows of this csv file:
"""

data = pd.read_csv("/scratch/w/wluyliu/yananc/ner_datasetreference.csv", encoding='unicode_escape')
data.head()
data.count()

"""As we can see, there are approximately 48,000 sentences in the dataset, 
comprising more than 1 million words and tags (quite huge!). This corresponds to approximately 20 words per sentence. 

Let's have a look at the different NER tags, and their frequency: 
"""

print("Number of tags: {}".format(len(data.Tag.unique())))
frequencies = data.Tag.value_counts()
frequencies

"""There are 8 category tags, each with a "beginning" and "inside" variant, and the "outside" tag. 
It is not really clear what these tags mean - "geo" probably stands for geographical entity, 
"gpe" for geopolitical entity, and so on.
 They do not seem to correspond with what the publisher says on Kaggle.
  Some tags seem to be underrepresented. Let's print them by frequency (highest to lowest): """

tags = {}
for tag, count in zip(frequencies.index, frequencies):
    if tag != "O":
        if tag[2:5] not in tags.keys():
            tags[tag[2:5]] = count
        else:
            tags[tag[2:5]] += count
    continue

print(sorted(tags.items(), key=lambda x: x[1], reverse=True))

"""Let's remove "art", "eve" and "nat" named entities, 
as performance on them will probably be not comparable to the other named entities. """

entities_to_remove = ["B-art", "I-art", "B-eve", "I-eve", "B-nat", "I-nat"]
data = data[~data.Tag.isin(entities_to_remove)]
data.head()

"""We create 2 dictionaries: 
one that maps individual tags to indices, 
and one that maps indices to their individual tags. 
This is necessary in order to create the labels 
(as computers work with numbers = indices, rather than words = tags) - see further in this notebook."""

labels_to_ids = {k: v for v, k in enumerate(data.Tag.unique())}
ids_to_labels = {v: k for v, k in enumerate(data.Tag.unique())}
labels_to_ids

"""As we can see, there are now only 10 different NER tags.

Now, we have to ask ourself the question: what is a training example in the case of NER, 
which is provided in a single forward pass? 

A training example is typically a **sentence**, with corresponding IOB tags.
 Let's group the words and corresponding tags by sentence:
"""

# pandas has a very handy "forward fill" function to fill missing values based on the last upper non-nan value
data = data.fillna(method='ffill')
data.head()

# let's create a new column called "sentence" which groups the words by sentence 
data['sentence'] = data[['Sentence #','Word','Tag']].groupby(['Sentence #'])['Word'].transform(lambda x: ' '.join(x))
# let's also create a new column called "word_labels" which groups the tags by sentence 
data['word_labels'] = data[['Sentence #','Word','Tag']].groupby(['Sentence #'])['Tag'].transform(lambda x: ','.join(x))
data.head()

"""Let's only keep the "sentence" and "word_labels" columns, and drop duplicates:"""

data = data[["sentence", "word_labels"]].drop_duplicates().reset_index(drop=True)
data.head()

len(data)

"""Let's verify that a random sentence and its corresponding tags are correct:"""

data.iloc[41].sentence

data.iloc[41].word_labels

"""#### **Preparing the dataset and dataloader**

Now that our data is preprocessed, we can turn it into PyTorch tensors such that we can provide it to the model. Let's start by defining some key variables that will be used later on in the training/evaluation process:
"""

MAX_LEN = 64
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
LEARNING_RATE = 1e-05
MAX_GRAD_NORM = 10
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased',\
     cache_dir='/scratch/w/wluyliu/yananc/cache', local_files_only=True)

"""A tricky part of NER with BERT is that BERT relies on **wordpiece tokenization**, rather than word tokenization. 
This means that we should also define the labels at the wordpiece-level, rather than the word-level! 

For example, if you have word like "Washington" which is labeled as "b-gpe", 
but it gets tokenized to "Wash", "##ing", "##ton", 
then one approach could be to handle this by only train the model on the tag labels for the first word piece token of a word 
(i.e. only label "Wash" with "b-gpe").
 This is what was done in the original BERT paper, see Github discussion [here](https://github.com/huggingface/transformers/issues/64#issuecomment-443703063).

Note that this is a **design decision**. 
You could also decide to propagate the original label of the word to all of its word pieces 
and let the model train on this. 
In that case, the model should be able to produce the correct labels for each individual wordpiece.
 This was done in [this NER tutorial with BERT](https://github.com/chambliss/Multilingual_NER/blob/master/python/utils/main_utils.py#L118). Another design decision could be to give the first wordpiece of each word the original word label, and then use the label “X” for all subsequent subwords of that word. All of them seem to lead to good performance.

Below, we define a regular PyTorch [dataset class](https://pytorch.org/docs/stable/data.html) 
(which transforms examples of a dataframe to PyTorch tensors). 
Here, each sentence gets tokenized, the special tokens that BERT expects are added, 
the tokens are padded or truncated based on the max length of the model,
 the attention mask is created and the labels are created based on the dictionary which we defined above.
  Word pieces that should be ignored have a label of -100
   (which is the default `ignore_index` of PyTorch's [CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)).

For more information about BERT's inputs, see [here](https://huggingface.co/transformers/glossary.html). 


"""

class dataset(Dataset):
  def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

  def __getitem__(self, index):
        # step 1: get the sentence and word labels 
        sentence = self.data['sentence'][index].strip().split()  
        word_labels = self.data['word_labels'][index].split(",") 

        # step 2: use tokenizer to encode sentence (includes padding/truncation up to max length)
        # BertTokenizerFast provides a handy "return_offsets_mapping" functionality for individual tokens
        encoding = self.tokenizer(sentence,
                             #is_pretokenized=True, 
                             is_split_into_words=True,
                             return_offsets_mapping=True, 
                             padding='max_length', 
                             truncation=True, 
                             max_length=self.max_len)
        
        # step 3: create token labels only for first word pieces of each tokenized word
        labels = [labels_to_ids[label] for label in word_labels] 
        # code based on https://huggingface.co/transformers/custom_datasets.html#tok-ner
        # create an empty array of -100 of length max_length
        encoded_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100
        
        # set only labels whose first offset position is 0 and the second is not 0
        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
          if mapping[0] == 0 and mapping[1] != 0:
            # overwrite label
            if i >= len(labels):
                encoded_labels[idx] = 0
            else: 
                encoded_labels[idx] = labels[i]
            i += 1

        # step 4: turn everything into PyTorch tensors
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.as_tensor(encoded_labels)
        
        return item

  def __len__(self):
        return self.len

"""Now, based on the class we defined above, we can create 2 datasets, 
one for training and one for testing. Let's use a 80/20 split:"""
from sklearn.model_selection import train_test_split

train_dataset, test_dataset = train_test_split(data, test_size=0.1)
train_dataset = train_dataset.reset_index(drop=True)
test_dataset = test_dataset.reset_index(drop=True)

test_dataset['sentence'] = test_dataset['sentence'].map(lambda x: x.replace('\xa0', ' '))


print("FULL Dataset: {}".format(data.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))

training_set = dataset(train_dataset, tokenizer, MAX_LEN)
testing_set = dataset(test_dataset, tokenizer, MAX_LEN)

"""Let's have a look at the first training example:"""

training_set[0]

"""Let's verify that the input ids and corresponding targets are correct:"""

for _ in range(7):
    sample_ix = random.sample(range(len(testing_set)), 1)[0]
    print("testcase:", sample_ix)
    for token, label in zip(tokenizer.convert_ids_to_tokens(testing_set[sample_ix]["input_ids"]), testing_set[sample_ix]["labels"]):
        print('{0:10}  {1} '.format(token, label), ids_to_labels.get(label.numpy().reshape(-1)[0], 'NOT_LABEL') )



"""Now, let's define the corresponding PyTorch dataloaders:"""

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

"""#### **Defining the model**

Here we define the model, BertForTokenClassification, and load it with the pretrained weights of "bert-base-uncased". 
The only thing we need to additionally specify is the number of labels (as this will determine the architecture of the classification head).

Note that only the base layers are initialized with the pretrained weights. 
The token classification head of top has just randomly initialized weights, which we will train, 
together with the pretrained weights, using our labelled dataset.
 This is also printed as a warning when you run the code cell below.

Then, we move the model to the GPU.
"""

model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(labels_to_ids),\
       cache_dir='/scratch/w/wluyliu/yananc/cache', local_files_only=True)
model.to(device)

"""#### **Training the model**

Before training the model, let's perform a sanity check, which I learned thanks to Andrej Karpathy's wonderful 
[cs231n course](http://cs231n.stanford.edu/) at Stanford (see also his [blog post about debugging neural networks](http://karpathy.github.io/2019/04/25/recipe/)). 
The initial loss of your model should be close to -ln(1/number of classes) = -ln(1/17) = 2.83. 

Why? Because we are using cross entropy loss. 
The cross entropy loss is defined as -ln(probability score of the model for the correct class). 
In the beginning, the weights are random, so the probability distribution 
for all of the classes for a given token will be uniform, 
meaning that the probability for the correct class will be near 1/17. 
The loss for a given token will thus be -ln(1/17). 
As PyTorch's [CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) 
(which is used by `BertForTokenClassification`) uses *mean reduction* by default, 
it will compute the mean loss for each of the tokens in the sequence for which a label is provided. 

Let's verify this:
"""

inputs = training_set[2]
input_ids = inputs["input_ids"].unsqueeze(0).to(device)
attention_mask = inputs["attention_mask"].unsqueeze(0).to(device)
labels = inputs["labels"].unsqueeze(0).to(device)


outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
initial_loss = outputs[0]
initial_loss

"""This looks good. 
Let's also verify that the logits of the neural network have a shape of (batch_size, sequence_length, num_labels):"""

tr_logits = outputs[1]
tr_logits.shape

"""Next, we define the optimizer. 
Here, we are just going to use Adam with a default learning rate. 
One can also decide to use more advanced ones such as AdamW (Adam with weight decay fix),
 which is [included](https://huggingface.co/transformers/main_classes/optimizer_schedules.html) 
 in the Transformers repository, and a learning rate scheduler, 
 but we are not going to do that here."""

optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

"""Now let's define a regular PyTorch training function. 
It is partly based on [a really good repository about multilingual NER](https://github.com/chambliss/Multilingual_NER/blob/master/python/utils/main_utils.py#L344)."""

# Defining the training function on the 80% of the dataset for tuning the bert model
def train(epoch):
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    tr_preds, tr_labels = [], []
    # put model in training mode
    model.train()
    
    for idx, batch in enumerate(training_loader):

        ids = batch['input_ids'].to(device, dtype = torch.long)
        mask = batch['attention_mask'].to(device, dtype = torch.long)
        labels = batch['labels'].to(device, dtype = torch.long)

        output_batch = model(input_ids=ids, attention_mask=mask, labels=labels)
        loss = output_batch['loss']
        tr_logits = output_batch['logits']

        tr_loss += loss.item()
         
        nb_tr_steps += 1
        nb_tr_examples += labels.size(0)
        
        if idx % 128 == 0:
            loss_step = tr_loss/nb_tr_steps
            print(f"Training loss per 128 training steps: {loss_step}")
           
        # compute training accuracy
        flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
        active_logits = tr_logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
        
        # only compute accuracy at active labels
        active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
        #active_labels = torch.where(active_accuracy, labels.view(-1), torch.tensor(-100).type_as(labels))
        
        labels = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)
        
        tr_labels.extend(labels.detach().cpu())
        tr_preds.extend(predictions.detach().cpu())

        tmp_tr_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
        tr_accuracy += tmp_tr_accuracy
    
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=MAX_GRAD_NORM
        )
        
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = tr_loss / nb_tr_steps
    tr_accuracy = tr_accuracy / nb_tr_steps
    print(f"Training loss epoch: {epoch_loss}")
    print(f"Training accuracy epoch: {tr_accuracy}")

    precision = precision_score(tr_labels, tr_preds, average='micro')
    recall = recall_score(tr_labels, tr_preds, average='micro')
    f1 = f1_score(tr_labels, tr_preds, average='micro')
    print("Training score:", precision, recall, f1)
    
"""And let's train the model!"""



"""#### **Evaluating the model**

Now that we've trained our model, we can evaluate its performance on the held-out test set 
(which is 20% of the data). Note that here, no gradient updates are performed, the model just outputs its logits.
"""

def valid(model, testing_loader):
    # put model in evaluation mode
    model.eval()
    
    eval_loss, eval_accuracy = 0, 0
    nb_eval_examples, nb_eval_steps = 0, 0
    eval_preds, eval_labels = [], []
    
    with torch.no_grad():
        for idx, batch in enumerate(testing_loader):

            ids = batch['input_ids'].to(device, dtype = torch.long)
            mask = batch['attention_mask'].to(device, dtype = torch.long)
            labels = batch['labels'].to(device, dtype = torch.long)
            
            output_batch = model(input_ids=ids, attention_mask=mask, labels=labels)
            loss = output_batch['loss']
            eval_logits = output_batch['logits']

            eval_loss += loss.item()

            nb_eval_steps += 1
            nb_eval_examples += labels.size(0)
        
            # compute evaluation accuracy
            flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
            active_logits = eval_logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
            
            # only compute accuracy at active labels
            active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
        
            labels = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)
            
            eval_labels.extend(labels)
            eval_preds.extend(predictions)
            
            tmp_eval_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
            eval_accuracy += tmp_eval_accuracy

    labels = [ids_to_labels[id.item()] for id in eval_labels]
    predictions = [ids_to_labels[id.item()] for id in eval_preds]
    
    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_steps
    print(f"Validation Loss: {eval_loss}")
    print(f"Validation Accuracy: {eval_accuracy}")
    precision = precision_score(labels, predictions, average='micro')
    recall = recall_score(labels, predictions, average='micro')
    f1 = f1_score(labels, predictions, average='micro')
    print("Validation score:", precision, recall, f1)


"""As we can see below, performance is quite good! Accuracy on the test test is > 93%."""



for epoch in range(1):
    print(f"Training epoch: {epoch + 1}")
    train(epoch)
    valid(model, testing_loader)






"""However, the accuracy metric is misleading, as a lot of labels are "outside" (O), 
even after omitting predictions on the [PAD] tokens. 
What is important is looking at the precision, recall and f1-score of the individual tags. 
For this, we use the seqeval Python library: """



  

"""Performance already seems quite good, but note that we've only trained for 1 epoch. 
An optimal approach would be to perform evaluation on a validation set while training to improve generalization.

#### **Inference**

The fun part is when we can quickly test the model on new, unseen sentences. 
Here, we use the prediction of the **first word piece of every word** (which is how the model was trained). 

*In other words, the code below does not take into account when predictions of different word pieces 
that belong to the same word do not match.*
"""



def infer(sentence, model):
    inputs = tokenizer(sentence.split(),
                        # is_pretokenized=True, 
                        is_split_into_words=True, 
                        return_offsets_mapping=True, 
                        padding='max_length', 
                        truncation=True, 
                        max_length=MAX_LEN,
                        return_tensors="pt")

    # move to gpu
    ids = inputs["input_ids"].to(device)
    mask = inputs["attention_mask"].to(device)
    # forward pass
    outputs = model(ids, attention_mask=mask)
    logits = outputs[0]

    active_logits = logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
    flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size*seq_len,) - predictions at the token level

    tokens = tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
    token_predictions = [ids_to_labels[i] for i in flattened_predictions.cpu().numpy()]
    wp_preds = list(zip(tokens, token_predictions)) # list of tuples. Each tuple = (wordpiece, prediction)

    prediction = []
    for token_pred, mapping in zip(wp_preds, inputs["offset_mapping"].squeeze().tolist()):
      #only predictions on first word pieces are important
      if mapping[0] == 0 and mapping[1] != 0:
        prediction.append(token_pred[1])
      else:
        continue

    dfr = pd.DataFrame(zip(sentence.split(), prediction), columns=['span','tag'])
    return dfr 


sent = "@HuggingFace is a company based in New York, but is also has employees working in Paris"

dfr = infer(sent, model)
print(dfr)


"""#### **Saving the model for future use**

Finally, let's save the vocabulary (.txt) file, model weights (.bin) and the model's configuration (.json) to a directory, so that both the tokenizer and model can be re-loaded using the `from_pretrained()` class method.
"""

import os

directory = "./model"

if not os.path.exists(directory):
    os.makedirs(directory)

# save vocabulary of the tokenizer
tokenizer.save_vocabulary(directory)
# save the model weights and its configuration file
model.save_pretrained(directory)







