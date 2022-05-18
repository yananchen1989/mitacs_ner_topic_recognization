import json
import requests
import pandas as pd 
from sklearn.model_selection import train_test_split

# API_TOKEN = "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"

# def query(payload='',parameters=None,options={'use_cache': False}):
#     API_URL = "https://api-inference.huggingface.co/models/EleutherAI/gpt-neo-2.7B"
#     headers = {"Authorization": f"Bearer {API_TOKEN}"}
#     body = {"inputs":payload,'parameters':parameters,'options':options}
#     response = requests.request("POST", API_URL, headers=headers, data= json.dumps(body))
#     try:
#       response.raise_for_status()
#     except requests.exceptions.HTTPError:
#         return "Error:"+" ".join(response.json()['error'])
#     else:
#       return response.json()[0]['generated_text']


from transformers import pipeline
from transformers import GPT2Tokenizer, GPT2LMHeadModel #TFGPT2LMHeadModel, TFGPT2Model, TFAutoModelForCausalLM

PATH_SCRATCH_CACHE = "/scratch/w/wluyliu/yananc/cache"
gen_nlp_gptneo = pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B', \
                    cache_dir=PATH_SCRATCH_CACHE, device=0, return_full_text=False)

tokenizer_gpt_neo = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-2.7B', cache_dir=PATH_SCRATCH_CACHE)


def query(prompt):
    input_tokens_cnt = len(tokenizer_gpt_neo(prompt)['input_ids'])
    result_gpt = gen_nlp_gptneo(prompt, max_length=input_tokens_cnt+16, \
                                        do_sample=True, temperature=0.4, # min_length=30,
                                        # top_p=0.9, top_k=0, \
                                        # repetition_penalty=1.2, num_return_sequences= 8,\
                                        clean_up_tokenization_spaces=True)
    # assert len(result_gpt) == 8
    contents_syn = [ii['generated_text'].strip() for ii in result_gpt if ii['generated_text'] ]
    return contents_syn[0]


# df_ft_train = get_ft('train')
# df_ft_dev = get_ft('dev')
# df_ft_test = get_ft('test')


# df_few_nerd = pd.concat([df_ft_train, df_ft_dev, df_ft_test])
# df_few_nerd.to_csv("/scratch/w/wluyliu/yananc/df_nerd.csv", index=False)

df_few_nerd = pd.read_csv("/scratch/w/wluyliu/yananc/df_nerd.csv")


for ix, row in  df_few_nerd.sample(100).iterrows():
    print(row['sent'])
    print(row['tag'])
    print(row['span'])
    print()


df = pd.read_csv("/scratch/w/wluyliu/yananc/sentence_level_annotations.csv")

def map_tag(x):
    if x in {'gov':'government', 'lab':'laboratory'}:
        return {'gov':'government', 'lab':'laboratory'}.get(x)
    else:
        return x 

df['tag'] = df['tag'].map(lambda x: map_tag(x))

demo_cnt = 10
# parameters = {
#     'max_new_tokens':30,  # number of generated tokens
#     'temperature': 0.1,   # controlling the randomness of generations
#     'end_sequence': "###" # stopping sequence for generation
# }


recall = []
precision = []

ite = 0
while 1:
    for tag in df['tag'].unique():
        if tag == '###':
            continue
        dft = df.loc[df['tag']==tag]
        dfts = dft.sample(demo_cnt+1)
        dft_in = dfts[:demo_cnt]
        dft_out = dfts[demo_cnt:]
        demos = []
        for ix, row in dft_in.iterrows():
            demo = "C: {} \nQ: Which is {} \nA: {}\n###\n".format(row['sent'], row['tag'], row['span'], tokenizer_gpt_neo.eos_token)
            demos.append(demo)
        context = ''.join(demos)

        demo = "C: {} \nQ: Which is {} \nA: ".format(dft_out['sent'].tolist()[0], dft_out['tag'].tolist()[0])
        prompt = context + demo
        print(prompt)







        for ix, row in dft_out.iterrows():
            
            prompt = context + demo
            response = query(prompt)
            #print(prompt)
            response_tokens = [ii.strip() for ii in\
             response.replace(prompt, '').replace(tokenizer_gpt_neo.eos_token,'').replace('_','').split(';') if ii.strip()]
            
            print(response_tokens)

            ref_tokens = [j.strip() for j in row['span'].split(';')]
            print( ref_tokens, '\n')

            for tt in response_tokens:
                if tt in ref_tokens:
                    precision.append(1)
                else:
                    precision.append(0)

            for tt in ref_tokens:
                if tt in response_tokens:
                    recall.append(1)
                else:
                    recall.append(0)
    
    ite += 1 
    if ite % 10 == 0:
        print(sum(precision) / len(precision), sum(recall) / len(recall))




