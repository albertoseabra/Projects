# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 14:03:18 2018

@author: Alberto Seabra
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import re


def tokenizing(text):
   # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
   tokens = [word for sent in sent_tokenize(text) for word in word_tokenize(sent)]
   filtered_tokens = []
   # filter out any tokens not containing letters (e.g., raw punctuation, numbers)
   for token in tokens:
       if re.search('[a-zA-Z]', token):
           filtered_tokens.append(token)
           
   return filtered_tokens


def build_tfidf(corpus):
    """
    Creates, fits to the corpus, and return the tf-idf vector from sklearn
    """
    tf = TfidfVectorizer(max_features=50000, stop_words='english', tokenizer=tokenizing, encoding='latin')
    tf.fit(corpus)
    
    return tf


def tfidf_sentences(tfidf_vector, text):
    """
    creates a list of sentences and tranforms them using the tf-idf vector
    returns the list of sentences and the tranformed vectores of the sentences
    """
    
    #getting the sentences:
    #TODO improve the sentences separator
    sentences = sent_tokenize(text)

    #tranforming the sentences using tf-idf
    text_vect = tfidf_vector.transform(sentences)
    
    return sentences, text_vect


def tfidf_summarizer(tfidf_vector, text, number_of_sentences):
    """
    Calls the function tfidf_sentences to get the sentences and the transformed vectores
    calculates the total importance of each sentence and divides by the number of words
    Sorts and prints the most important sentences by the order they appear in the text
    """
    
    sentences, text_vect = tfidf_sentences(tfidf_vector, text)
    
    #iterate for each vector, get the sum of tf-idf and then divide by the number of words
    sentence_importance = []
    for i in range(text_vect.shape[0]):
        soma = text_vect[i].sum()
        words = text_vect[i].count_nonzero()
        #in case a sentence has only stopwords and/or numbers the sum of tf-idf will be zero
        #we give it the importance value of 0
        if soma == 0:
            sentence_importance.append(0)
        #in case is a really short sentence it might get high importance value
        #but it wont give us much information, going to ignore them
        elif words < 5:
            sentence_importance.append(0)
        else:
            sentence_importance.append(soma/words)
        
    #sorting the importance of the sentences in descending order
    sorted_importance = sorted(sentence_importance, reverse=True)
    
    # getting the index of the sentences in the summary
    order = []
    for i in range(number_of_sentences): 
        first = sentence_importance.index(sorted_importance[i])
        order.append(first)
     
    #printing the sentences following the order in the text
    for i in range(number_of_sentences):
        first = min(order)
        print(sentences[first])
        order.remove(first)
        



"""
here starts text rank summarizer, based on graphs and PageRank
"""

def create_graph(tfidf_vector, text):
    """
    Creates a graph for the text with the sentences as nodes and the cosine similarity 
    between the sentences as the edges
    returns the graph and a list with the sentences
    """
    #initialize networkx graph
    graph = nx.Graph()
    
    #getting the sentences and the tfidf vectores od the sentences
    sentences, text_vect = tfidf_sentences(tfidf_vector, text)
    
    #iterates of each pair of sentences
    size = len(sentences)
    for index1 in range(size):
        for index2 in range(index1 + 1,size):
            if index1 == index2:
                continue
            #calculates the similarity between the pair of sentences
            #creates and graph edge between the sentences with the similarity as edge weight
            else:
                graph.add_edge(index1, index2, 
                               weight= cosine_similarity(text_vect[index1].toarray(),
                                                         text_vect[index2].toarray()))
    
    return graph, sentences     
        


def textrank_summarizer(tfidf_vector, text, number_sentences):
    """
    Calls the function create_graph to create the graph and calculates the Pagerank value
    of the nodes. Sorts the sentences in descending order of importance
    Prints the most important sentences, depending of the number_sentences
    
    """
    #creating the graph
    graph, sentences = create_graph(tfidf_vector, text)
    
    #calculate Pagerank
    rank = nx.pagerank(graph, weight='weight')
    
    #sorting by Rank value
    sort = sorted(rank, key=rank.get, reverse=True)
    
    #Printing the sentences with highest rank
    sentences_index = sort[:number_sentences]
    for i in sentences_index:
        print (sentences[i])   



df = pd.read_csv(r'C:\bts_master\project\news\data.csv', encoding= 'latin')
corpus = df.text.values

#CREATE THE TF-IDF VECTOR BASED ON A CORPUS:
tfidf_vector = build_tfidf(corpus)

#SUMMARIZE, text is the text to summarize and the number of sentences is how many sentences you want
#textrank_summarizer(tfidf_vector, text, number_sentences)
#or
#tfidf_summarizer(tfidf_vector, text, number_of_sentences)
