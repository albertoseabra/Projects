# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 09:13:04 2018

@author: Alberto Seabra
"""

import requests
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize


page = requests.get(url)

if page.status_code == requests.codes.ok:
    
    soup = BeautifulSoup(page.content, 'lxml')
    
    title = soup.find('h1').text
    
    # find the text
    text = [line.text for line in soup.findAll('p')]
    
    #using this will return us each sentence, sometimes those sentences are very short
    # and its really hard to understand what's going on with a short sentence
    #tokenize each paragraph and flatten the list of lists
    text_sentences = [sent_tokenize(line) for line in text]
    sentences = [sentence for sublist in text_sentences for sentence in sublist]
    
    

else:
    print('Something went wrong while getting the webpage')