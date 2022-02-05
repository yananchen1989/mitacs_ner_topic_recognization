# python
# coding: utf-8

######################################
##########  Changes to Make  #########
######################################

# Include Column for Search Query + Verctorize Search Query to Loop
# Filter URL to search within quant-ph only, etc
# Can add .replace('\n', ' ') to abstract to remove line breaks... ideal?

######################################
########## Library/Packages ##########
######################################

#!pip install feedparser

import pandas as pd
import numpy as np

import urllib
import time
import feedparser
from IPython.display import display

######################################
########## Calling from API ##########
######################################

# Base API Query URL
base_url = 'http://export.arxiv.org/api/query?'

# Search Parameters
search_query = 'all:quantum%20computing'    # search for articles relating to quantum computing
start = 0                                   # start at the first result
total_results = 4                          # want # total results (suggested was 20)
results_per_iteration = 2                   # Number of results at a time (suggested was 5)
wait_time = 1                               # number of seconds to wait between calls (suggested was 3)

#Pre-Set
Cnt=0
API_Dict={}

print('Searching arXiv for: "%s" ' % search_query)


#Looping Through Results
for i in range(start,total_results,results_per_iteration):    
    
    #Output Result #s
    print("Results %i - %i" % (i,i+results_per_iteration))
    
    #Query Term Using Inputs
    query = 'search_query=%s&start=%i&max_results=%i' % (search_query, i, results_per_iteration)
    
    #Request from ArXiv Using URL+Query to form request
    print('%s' % base_url+query)
    response = urllib.request.urlopen(base_url+query).read()
    
    #Parse the response using feedparser
    feed = feedparser.parse(response)

    # Run through each entry, and grab selected information
    for entry in feed.entries:
        #Enter Values into Dictionary
        API_Dict[Cnt] = [entry.id, entry.title, entry.published[:10], entry.author, entry.summary]
        Cnt += 1

    #Wait to re-call from API
    print('Wait %i Second(s)' % wait_time)
    time.sleep(wait_time)
    
# Exited For Loop
print("Data Gathering Complete")

# Generate Dataframe, Make Edits, Sort, Output to CSV
API_Output = pd.DataFrame.from_dict(API_Dict,orient='index', columns = ["URL", "Title", "Published", "Author", "Abstract"])
API_Output["Published"] = pd.to_datetime(API_Output["Published"])
API_Output = API_Output.sort_values(by="Published", ascending=False)

API_Output.to_csv('nlp4quantumpapers/datasets/ArXiv/API_Output.csv',index=False)
print("Output to CSV Complete")

