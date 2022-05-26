
# # Conditional Generation with Prefix Tuning.
# In this tutorial, we do conditional generation with prefix tuning template.

# we use WebNLG as an example, as well. Note that the evaluation of generation result should be done
# by using the scripts provided by https://github.com/Yale-LILY/dart/tree/master/evaluation, 
# Which we do not include in it. 

import argparse
import torch,os

parser = argparse.ArgumentParser("")
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--plm_eval_mode", action="store_true")
parser.add_argument("--model", type=str, default='t5-base')  # tested model are gpt2/t5
parser.add_argument("--freeze_plm", action="store_true")
parser.add_argument("--template", type=str)
parser.add_argument("--source_col", type=str)
parser.add_argument("--gpu", default="", type=str)
args = parser.parse_args()


print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# from openprompt.data_utils.conditional_generation_dataset import WebNLGProcessor
# dataset = {}
# dataset['train'] = WebNLGProcessor().get_train_examples("/home/yanan/OpenPrompt/datasets/CondGen/webnlg_2017/")
# dataset['validation'] = WebNLGProcessor().get_dev_examples("/home/yanan/OpenPrompt/datasets/CondGen/webnlg_2017/")
# dataset['test'] = WebNLGProcessor().get_test_examples("/home/yanan/OpenPrompt/datasets/CondGen/webnlg_2017/")

'''
{
  "guid": "11",
  "label": null,
  "meta": {},
  "text_a": " | Aarhus_Airport : operatingOrganisation : Aktieselskab",
  "text_b": "",
  "tgt_text": "Aktieselskab operates Aarhus Airport."
}
'''

# import datasets
# ds2020 = datasets.load_dataset('web_nlg', 'release_v3.0_en')
# ds2017 = datasets.load_dataset('web_nlg', 'webnlg_challenge_2017')





from sklearn.model_selection import train_test_split
from utils.load_data import * 
#ds = load_data(dataset='ag', samplecnt= 128)

#ds.df_train['content'] = ds.df_train['content'].map(lambda x: remove_str(x))

df = pd.read_csv("./torch_ds/df_cc_news_ners.csv", lineterminator='\n')

dfs = df.sample(10000)

dfs = dfs.loc[(~dfs['ners'].isnull()) & (dfs['ners'].str.contains('<=>')) & \
                (~dfs['title'].isnull()) &  \
                 (~dfs['content'].isnull()) ]

dfs['ners'] = dfs['ners'].map(lambda x: ', '.join(list(set(x.split('<=>')))))

df_train, df_test = train_test_split(dfs, test_size=0.15)


from openprompt.data_utils import InputExample



dataset = {}
dataset['train'] = []
dataset['test'] = []

for ix, row in df_train.reset_index().iterrows():
    dd = InputExample(
        guid = str(ix),
        tgt_text = row['content'],
        text_a = row[args.source_col],
        #text_b = row['content']
    )
    dataset['train'].append(dd)

# df_test = ds.df_test.sample(1024)
for ix, row in df_test.reset_index().iterrows():
    dd = InputExample(
        guid = str(ix),
        tgt_text = row['content'],
        text_a = row[args.source_col],
        #text_b = row['content']
    )
    dataset['test'].append(dd)


# load a pretrained model, its tokenizer, its config, and its TokenzerWrapper by one function 
from openprompt.plms import load_plm
plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model, "./cache")

# Instantiating the PrefixTuning Template !

if args.template == 'prefix':
    from openprompt.prompts.prefix_tuning_template import PrefixTuningTemplate
    # we can use a plain text as the default setting
    # i.e. 
    # mytemplate = PrefixTuningTemplate(model=plm, tokenizer=tokenizer)
    # is equal to 
    # mytemplate = PrefixTuningTemplate(model=plm, tokenizer=tokenizer, text='{"placeholder":"text_a"} {"mask"}')
    mytemplate = PrefixTuningTemplate(model=plm,  tokenizer=tokenizer, \
                    text='This is a {"placeholder":"text_a"} News: {"special": "<eos>"} {"mask"} ', \
                     # {"placeholder":"text_b"}
                        using_decoder_past_key_values=False, num_token=5)

elif args.template == 'soft':
    from openprompt.prompts import SoftTemplate
    mytemplate = SoftTemplate(model=plm, tokenizer=tokenizer,   \
             #text='{"placeholder":"text_a"} {"soft"} {"soft"} {"soft"} {"placeholder":"text_b"} {"soft"} {"mask"}.'
            text='{"placeholder":"text_a"} {"soft"} {"soft"} {"soft"} {"soft"} {"mask"}.'
             )

elif args.template == 'mixed':
    from openprompt.prompts import MixedTemplate
    mytemplate = MixedTemplate(model=plm, tokenizer=tokenizer,\
         #text='{"placeholder":"text_a"} {"soft": "Question:"} {"placeholder":"text_b"}? Is it correct? {"soft"} {"mask"}.')
    text='{"placeholder":"text_a"} {"soft": "Breaking News:"}. Another related News is {"soft"} {"mask"}.')

elif args.template == 'mannual':
    from openprompt.prompts import ManualTemplate
    mytemplate = ManualTemplate(tokenizer=tokenizer, text='This is a {"placeholder":"text_a"} News: {"mask"}.')

#elif args.template == 'p':
    



# To better understand how does the template wrap the example, we visualize one instance.
# You may observe that the example doesn't end with <|endoftext|> token. Don't worry, adding specific end-of-text token
# is a language-model-specific token. we will add it for you in the TokenizerWrapper once you pass `predict_eos_token=True`
wrapped_example = mytemplate.wrap_one_example(dataset['train'][0]) 
print(wrapped_example)


# Your can loop over the dataset by yourself by subsequently call mytemplate.wrap_one_example  and WrapperClass().tokenizer()
# but we have provide a PromptDataLoader for you.
from openprompt import PromptDataLoader
train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer, 
    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=256, 
    batch_size=16,shuffle=True, teacher_forcing=True, predict_eos_token=True, # be sure to pass predict_eos_token=True if your tempalte doesn't contain one, or you model may fail to stop generation.
    truncate_method="head")


test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer, 
    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=256, 
    batch_size=16,shuffle=False, teacher_forcing=False, predict_eos_token=True,
    truncate_method="head")

# load the pipeline model PromptForGeneration.
from openprompt import PromptForGeneration

prompt_model = PromptForGeneration(plm=plm,template=mytemplate, freeze_plm=args.freeze_plm,tokenizer=tokenizer, plm_eval_mode=args.plm_eval_mode).cuda()


from transformers import AdamW
# Follow PrefixTuning（https://github.com/XiangLi1999/PrefixTuning), we also fix the language model
# only include the template's parameters in training. 

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
{
    "params": [p for n, p in mytemplate.named_parameters() if (not any(nd in n for nd in no_decay)) and p.requires_grad],
    "weight_decay": 0.0,
},
{
    "params": [p for n, p in mytemplate.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
    "weight_decay": 0.0,
},
]


optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)

from transformers.optimization import get_linear_schedule_with_warmup

tot_step  = len(train_dataloader)*5
scheduler = get_linear_schedule_with_warmup(optimizer, 0, tot_step)

# We provide generation a generation metric, you can also define your own. Note that it's not directly comparable to WebNLG's scripts evaluation.
from openprompt.utils.metrics import generation_metric
# Define evaluate function 
def evaluate(prompt_model, dataloader):
    generated_sentence = []
    groundtruth_sentence = []
    prompt_model.eval()

    for step, inputs in enumerate(dataloader):
        _, output_sentence = prompt_model.generate(inputs.cuda(), **generation_arguments)
        generated_sentence.extend(output_sentence)
        groundtruth_sentence.extend(inputs['tgt_text'])
        if step > 3:
            break 
        #labels.extend(inputs['label_name'])
    #score = generation_metric(generated_sentence, groundtruth_sentence, "sentence_bleu")
    #print("test_score", score, flush=True)
    return generated_sentence, groundtruth_sentence




generation_arguments = {
    "max_length": 128,
    "max_new_tokens": None,
    "min_length": 32,
    "temperature": 1.0,
    "do_sample": False,
    "top_k": 0,
    "top_p": 0.9,
    "repetition_penalty": 1.2,
    "num_beams": 5,
    "bad_words_ids": [[628], [198]]
}

# training and generation.

for epoch in range(25):
    prompt_model.train()
    epoch_loss = []
    for step, inputs in enumerate(train_dataloader):
        loss = prompt_model(inputs.cuda())
        loss.backward()
        epoch_loss.append( loss.item()/16 )
        torch.nn.utils.clip_grad_norm_(mytemplate.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    print("train epoch:", epoch, "loss:", sum(epoch_loss) /len(epoch_loss) )


    if epoch >= 3:
        generated_sentence_train, groundtruth_sentence_train = evaluate(prompt_model, train_dataloader)

        print('train set:')
        for ii in random.sample(list(zip( generated_sentence_train, groundtruth_sentence_train)), 32):
            #print('label==>', ii[0])
            print('syn==>\n', ii[0])
            print('ref==>\n', ii[1])
            print('\n')
        print('\n\n')


        generated_sentence_test, groundtruth_sentence_test = evaluate(prompt_model, test_dataloader)
        print('test set:')
        for ii in random.sample(list(zip(generated_sentence_test, groundtruth_sentence_test)), 32):
            print('syn==>\n', ii[0])
            print('ref==>\n', ii[1])
            print('\n')
        print('\n\n')

