# python -u
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

# General Packages
import pandas as pd
import numpy as np

# API Required/Suggested
import urllib
import time
import feedparser

# For Clearing Terminal
import os
os.system("cls")

######################################
########## Calling from API ##########
######################################

# Setup Search Terms / Categories / Other Restrictions
search_terms = ['super%20conducting']
# search_terms = ['superconducting','super%20conducting','super+conducting',
#                 '"superconducting"','"super%20conducting"','"super+conducting"',
#                 'superconduct','super%20conduct','super+conduct',
#                 '"superconduct"','"super%20conduct"','"super+conduct"']
                
categories = ['cat:quant-ph']

#%28all:"violet"+AND+cat:quant-ph%29&start=0&max_results=10
#all: may not be necessary

# Base API Query URL
base_url = 'http://export.arxiv.org/api/query?search_query='

# Search Parameters
filter_cat = False                       # True or False --> Fix Query to ArXiv Categories (True = search specific categories)
start = 0                                   # start at the first result
total_results = 1600                          # want # total results (suggested was 20)
results_per_iteration = 20                   # Number of results at a time (suggested was 5)
wait_time = 3                               # number of seconds to wait between calls (suggested was 3)

#Pre-Set
Cnt=0
API_Dict={}

for search_term in search_terms:
    search_query = 'all:%s' % (search_term)    # search for articles relating to quantum computing

    print('Searching arXiv for: "%s" ' % search_query)

    if filter_cat == True:
        filter = '+AND+%s' % (categories[0])
    else:
        filter = ''

    #Looping Through Results
    for i in range(start,total_results,results_per_iteration):    
        
        #Output Result #s
        print("Results %i - %i" % (i,i+results_per_iteration))
        
        #Query Term Using Inputs
        results = '%%29&start=%i&max_results=%i' % (i, results_per_iteration)
        query = '%%28%s' % (search_query+filter+results)

        #Request from ArXiv Using URL+Query to form request
        print('%s' % base_url+query)
        response = urllib.request.urlopen(base_url+query).read()
        
        #Parse the response using feedparser
        feed = feedparser.parse(response)

        #print(feed.entries)
        #if feed.entries == []: break # --> Exit feed if no entries - need to figure out correct method this does not work currently

        # Run through each entry, and grab selected information
        for entry in feed.entries:
            #Enter Values into Dictionary
            API_Dict[Cnt] = [entry.id, entry.title, entry.published[:10], entry.author, entry.summary, entry.category, categories[0], search_query]
            Cnt += 1

        #Wait to re-call from API
        print('Wait %i Second(s)' % wait_time)
        time.sleep(wait_time)

# Exited For Loop
print("Data Gathering Complete")

# Generate Dataframe, Make Edits, Sort, Output to CSV
API_Output = pd.DataFrame.from_dict(API_Dict,orient='index', columns = ["URL", "Title", "Published", "Author", "Abstract", "Atom Category", "Filtered Category", "Search Term"])
API_Output["Published"] = pd.to_datetime(API_Output["Published"])
API_Output = API_Output.sort_values(by="Published", ascending=False)

if filter_cat == True: CSV_Path = 'nlp4quantumpapers/datasets/ArXiv/API_Output_CatFilterTrue_Testing.csv'
if filter_cat == False: CSV_Path = 'nlp4quantumpapers/datasets/ArXiv/API_Output_CatFilterFalse_Testing.csv'

#Temporarily Make General Single-Search Term Output
CSV_Path = 'nlp4quantumpapers/datasets/ArXiv/API_Output.csv'

API_Output.to_csv(CSV_Path,index=False)
print("Output to CSV Complete")

