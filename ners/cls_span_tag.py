import tensorflow as tf
import pandas as pd 
from tensorflow.keras import layers
from tensorflow.keras.callbacks import *
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.metrics import confusion_matrix
import numpy as np 
import random
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



def get_model_bert(num_classes):

    PATH_HOME = "/home/w/wluyliu/yananc/topic_classification_augmentation"

    preprocessor_file = "{}/resource/albert_en_preprocess_3".format(PATH_HOME) # https://tfhub.dev/tensorflow/albert_en_preprocess/3
    preprocessor_layer = hub.KerasLayer(preprocessor_file)

    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string) # shape=(None,) dtype=string

    encoder = hub.KerasLayer("{}/resource/albert_en_base_2".format(PATH_HOME), trainable=True, name=str(random.sample(list(range(10000)), 1)[0]))

    encoder_inputs = preprocessor_layer(text_input)
    outputs = encoder(encoder_inputs)
    embed = outputs["pooled_output"]  

    if num_classes == 2:
        out = layers.Dense(1, activation='sigmoid')(embed)
        model = tf.keras.Model(inputs=text_input, outputs=out)
        model.compile(Adam(learning_rate=2e-5), "binary_crossentropy", metrics=["binary_accuracy"])
    else:
        out = layers.Dense(num_classes, activation="softmax")(embed)
        model = tf.keras.Model(inputs=text_input, outputs=out)
        model.compile(Adam(learning_rate=2e-5), "sparse_categorical_crossentropy", metrics=["acc"])
    return model


df_tags = pd.read_csv("./datasets/QI-NERs.csv")

ixl = {i:j for i,j in enumerate(df_tags['tag'].drop_duplicates().tolist()) }
ixl_rev = {j:i for i,j in enumerate(df_tags['tag'].drop_duplicates().tolist()) }


df_tags['label'] = df_tags['tag'].map(lambda x: ixl_rev[x])

from sklearn.model_selection import train_test_split



df_train, df_test = train_test_split(df_tags, test_size=0.1)


x_train = df_train['span'].values.reshape(-1,1)
y_train = df_train['label'].values

x_test = df_test['span'].values.reshape(-1,1)
y_test = df_test['label'].values

x_all = df_tags['span'].values.reshape(-1,1)
y_all = df_tags['label'].values

model = get_model_bert(df_tags['label'].unique().shape[0])

model.fit(
        x_all, y_all, batch_size=8, epochs=5, \
        validation_data=(x_test, y_test), verbose=1,
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='acc', patience=3, mode='max',restore_best_weights=True)]
    )


model.save("cls_dict")

preds = model.predict(x_test, batch_size=64)
preds_label = preds.argmax(axis=1)

preds_tag = [ixl[i] for i in preds_label]
y_test_tag = [ixl[i] for i in y_test]

#Get the confusion matrix
cm = confusion_matrix(y_test_tag, preds_tag)
acc = cm.diagonal().sum() / cm.sum()
print(ite, acc)


pred = model.predict(['ColdQuanta'])

pred_tag = ixl[pred.argmax(axis=1)[0]]
pred_score = pred.max()



for i in np.unique(y_test):
    acc_class = cm.diagonal()[i] / cm[:,i].sum()
    print("acc_class==>", ixl[i], acc_class)










