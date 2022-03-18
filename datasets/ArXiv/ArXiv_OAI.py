# python -u
# coding: utf-8

######################################
########## Library/Packages ##########
######################################

#!pip install feedparser

# General Packages
from asyncio.windows_events import NULL
import pandas as pd
import numpy as np
from datetime import datetime

# API Required/Suggested
import urllib
import time
import feedparser
import xml.etree.ElementTree as ET
import xmltodict


# For Clearing Terminal
import os
os.system("cls")

######################################
########## Calling from OAI ##########
######################################

# Setup Search Terms / Categories / Other Restrictions
use_token = True
date_from = '2022-03-14'
category = 'physics:quant-ph'
pull_type = 'arXiv' #Can also use arXivOld, arXivRaw, oai_dc - but arXiv is chosen as best

wait_time = 5.1 #How long to pause between article pulls (site says 3s per request, but some issues - use 5+)
resume_token = ''
last_pull_start = datetime.now()

# Pre-Declare List
article_id_list = []

while resume_token != 'None':

    # OAI Query URL's
    if (use_token == True) and (resume_token == ''):
        search_url = 'https://export.arxiv.org/oai2?verb=ListIdentifiers&metadataPrefix=%s&set=%s' % (pull_type, category)
        print("Initialize Token Harvest with: %s" % (search_url))
    elif (use_token == True) and (resume_token != ''):
        search_url = 'https://export.arxiv.org/oai2?verb=ListIdentifiers&resumptionToken=%s' % (resume_token)
        print("Continue Token Harvest with: %s" % (search_url))
    elif (use_token == False):
        search_url = 'https://export.arxiv.org/oai2?verb=ListIdentifiers&from=%s&metadataPrefix=%s&set=%s' % (date_from, pull_type, category)
        print("Continue Harvest by From-Date with: %s" % (search_url))
    else:
        print("Case Unaccounted For")

    # Pull Info
    results = urllib.request.urlopen(search_url).read()

    #Added example of response catching - to be added in for potential 503 retry requests
    #retries = 1
    #success = False
    #while not success:
    #    try:
    #        response = urllib2.urlopen(request)
    #        success = True
    #    except Exception as e:
    #        wait = retries * 30;
    #        print 'Error! Waiting %s secs and re-trying...' % wait
    #        sys.stdout.flush()
    #        time.sleep(wait)
    #        retries += 1



    # Parse XML Format to Dict
    feed = xmltodict.parse(results)

    # Pull Desired Nodes (Gives Identifier, Datestamp, SetSpec -- "Article ID", "Date Added to OAI", "Category")
    feed_refined = feed['OAI-PMH']['ListIdentifiers']['header']
    feed_token = feed['OAI-PMH']['ListIdentifiers']['resumptionToken']

    # Note Total Size of List (For Cross-Validation that All Files Pulled Successfully)
    if (use_token == True):
        total_size = int(feed_token['@completeListSize'])
    else:
        total_size = len(feed_refined)

    if '#text' in feed_token:
        resume_token = feed_token['#text']
        print("Resume Token: %s" % (resume_token))
    else:
        resume_token = 'None'
        print("No Resume Token Found")

    # Loop through list of dicts with desired information, pull article ID to construct metadata URL
    for article_id in range(len(feed_refined)):
        article_id_list.append(feed_refined[article_id]['identifier'])
    
    # Wait Between URL Pull
    print('Wait %i Second(s) b/w Token URL Pull(s)\n' % wait_time)
    time.sleep(wait_time)


# Note Finish of Pull Step
print('Token URL Pull Finished\n')
print('OAI Size = %i' % (total_size))
print('Article Size = %i' % (len(article_id_list)))
print ("Cross-Check = VERIFIED" if (total_size == len(article_id_list)) else "Cross-Check = ERROR")

#Pre-Allocate
Results_Dict = {}
Cnt = 0

# Use article ID to pull metadata for specific article
print('\n\n\n')
print('Begin Article URL Pulls - IN PROGRESS (Updates Every ~2m)')

for article_id in article_id_list[-2]: #Can add [-4:] to pull last few entries for testing purposes
    meta_url = 'https://export.arxiv.org/oai2?verb=GetRecord&identifier=%s&metadataPrefix=arXiv' % (article_id)
    meta_results = urllib.request.urlopen(meta_url).read()
    meta_feed = xmltodict.parse(meta_results)


    #Isolate Authors into Single List, Break out Affiliations if any exist
    author_list = meta_feed['OAI-PMH']['GetRecord']['record']['metadata']['arXiv']['authors']['author']
    author_list_refined = ''
    affiliation = "None"

    if isinstance(author_list, list):
        for dict_entry in author_list:
            for k, v in dict_entry.items():
                if k != 'affiliation':
                    author_list_refined = '%s%s,' % (author_list_refined,v) # Gets Last, then Gets First
                else:
                    affiliation = v
            
            author_list_refined = '%s|' % (author_list_refined[:-1]) # Breaks for Next Author (if any)

    else:
        for k, v in author_list.items():
            if k != 'affiliation':
                author_list_refined = '%s%s,' % (author_list_refined,v)
            else:
                affiliation = v

    # Place into Dictionary
    Results_Dict[Cnt] = [
                        meta_feed['OAI-PMH']['GetRecord']['record']['metadata']['arXiv']['id'], #arxiv.org/abs/1311.0456
                        meta_feed['OAI-PMH']['GetRecord']['record']['metadata']['arXiv']['created'],
                        meta_feed['OAI-PMH']['GetRecord']['record']['metadata']['arXiv']['title'],
                        meta_feed['OAI-PMH']['GetRecord']['record']['metadata']['arXiv']['abstract'],
                        meta_feed['OAI-PMH']['GetRecord']['record']['metadata']['arXiv']['categories'],
                        author_list_refined[:-1],
                        affiliation
                        ]
                        #Category Options: 'id', 'created', 'updated', 'authors', 'title', 'categories', 'comments', 'journal-ref', 'doi', 'license', 'abstract'
                        #Note: Comments, Journal-Ref, DOI, License are not always included. Would need to add some statements to work around.

    Cnt += 1

    # Show Progress
    if (article_id_list.index(article_id) + 1) % 25 == 0:
        progress_rate = round((article_id_list.index(article_id) + 1)/len(article_id_list), 4)
        print('Progress: %d%% (Article %i of %i)' % (progress_rate*100, article_id_list.index(article_id), len(article_id_list)))
        print('Current Article Metadata URL: %s' % (meta_url))

    #Ensure we wait between URL calls
    time.sleep(wait_time)

# Note that Article URLs Have Been Harvested
print('Article URL Pull Finished\n')
print('Articles Pulled: %i' % (Cnt+1))

# Generate Dataframe, Make Edits, Sort, Output to CSV
OAI_Output = pd.DataFrame.from_dict(Results_Dict, orient='index', columns = ["ID", "Created", "Title", "Abstract", "Categories", "Authors", "Affiliation","Comments"])
OAI_Output["Created"] = pd.to_datetime(OAI_Output["Created"])
OAI_Output = OAI_Output.sort_values(by="Created", ascending=False)

OAI_Output.to_csv("Run Files Here/OAI_TK_Test1.csv", index=False)
print("Output to CSV Complete")

last_pull_end = datetime.now()

