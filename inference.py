import nltk
import json,random,string
import pandas as pd 
import numpy as np
# huggingface transformers
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

# this is where you save the model 
MODEL_PATH = "/scratch/w/wluyliu/yananc/finetunes/roberta_tqi/"
CONFIDENT = 0.8

# reload the model from the disk to the memory (gpu)
tokenizer_roberta_nerd_fine = AutoTokenizer.from_pretrained(MODEL_PATH)
model_roberta_nerd_fine = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
nlp_ner = pipeline("ner", model=model_roberta_nerd_fine, tokenizer=tokenizer_roberta_nerd_fine,\
                     aggregation_strategy="simple", device=1)

# nlp_ner will be used for inference


content = """
Applied Quantum, QC Ed, Net Optimization Gains, Classical vs Quantum, Hybrid Solvers and more . . . 
Relevant to: Potential Customers, Workforce, Learners, Higher Ed  Technology 
Q: How would you describe applied quantum, how D-Wave became focused on that, and how people can get started? 
D-Wave has been focused on practical quantum computing since the company’s inception. 
In the early years it was simply about being able to build a real, usable quantum computer. 
Today, it’s about helping customers build applications using real data to solve real problems with real business impact. 
The problems our customers are solving, therefore, have evolved over the years and instead of devoting most of our team’s resources to determining how we build quantum systems, we are now more focused on exploring how we continue to improve the performance and scale of our systems so that businesses can benefit. 
To help customers get started, we recently introduced a new jump-start program called D-Wave Launch. The program is for enterprises who are interested in developing hybrid quantum applications and could use additional support.
 In a nutshell, we help users identify the right problems (most often optimization problems) and then help them map the problem onto the quantum processor. 
 Collaboration typically proceeds in four phases: 1) help discover which business challenges are best suited for D-Wave’s hybrid quantum service, 2) work with the customer team to build a proof-of-concept, 3) pilot the hybrid quantum application, and 4) put the hybrid quantum application into production. 
 Q: How would you describe the most “doable” applications for D-Wave’s system today? 
 Are these uniquely tied to the “annealing” approach vs other methodologies? Annealing systems — the model of D-Wave’s processors — are particularly well-suited for optimization problems, though they are capable of solving an array of NP-hard problems. 
 Optimization problems are very widely applicable for a broad range of problem types and applications that have commercial value today. For example, it includes everything from optimizing manufacturing processes to retail operations or the allocation of hospital resources. For one example, Volkswagen used quantum computing to build an application that optimizes paint shop operations to reduce cost and waste. Long-term, quantum computers — including our own — are capable of solving diverse types of problems. Today, quantum annealers are demonstrating business value across many practical optimization problems while outperforming other quantum platforms. Q: How would you address skepticism about whether present generation quantum hardware is actually resulting in unique optimization gains? There will always be skeptics, but the good news is that the science doesn’t align with that skepticism. D-Wave users are already seeing early advantages in using a D-Wave system over existing methods to solve complex business problems. There’s that business gain that companies are starting to see with quantum computing, and there is promising research we’re working on too. We recently published a peer-reviewed paper in Nature Communications, marking a major milestone on the journey to quantum advantage. The new research uses a D-Wave lower noise system to demonstrate a 3 million times speed-up over classical alternatives in a real-world problem. This is the first time a speed-up of this kind — with scaling advantages in both temperature and size — 
has been demonstrated on a practically valuable problem with implications for the real world. 
"""      

def infer(content):
    # split content into sentences
    sents = nltk.sent_tokenize(content)

    infos = []
    for sent in sents:
        if len(sent) <= 5:
            continue
        res = nlp_ner(sent)
        if not res:
            continue
        for ii in res:
            # hyper parameter, filter out predictions of low confidence 
            # this can be changed based on practice
            if ii['score'] >= CONFIDENT: 
                infos.append((ii['entity_group'], ii['score'], ii['word']))

    df_res = pd.DataFrame(infos, columns=['label','score','entity'])

    df_res.sort_values('score', ascending=False, inplace=True)
    df_res.drop_duplicates(['entity'], inplace=True)
    return df_res

df_res = infer(content)

df_res.to_csv("df_res.csv", index=False)

'''
     label     score       entity
0  company  0.998435       D-Wave
6  company  0.984229   Volkswagen
'''
