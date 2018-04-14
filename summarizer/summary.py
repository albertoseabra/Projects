# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 14:03:18 2018

@author: Alberto Seabra
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import LancasterStemmer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import re
import pickle

stemmer = LancasterStemmer()

def tokenizing(text, stemming=True):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in sent_tokenize(text) for word in word_tokenize(sent)]
    filtered_words = []
    # filter out any tokens not containing letters (e.g., raw punctuation, numbers)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_words.append(token)
    # if we want to stemm the words:       
    if stemming:
        stemmed_words = [stemmer.stem(word) for word in filtered_words]
        return stemmed_words
    else:        
        return filtered_words


def build_tfidf(corpus):
    """
    Creates, fits to the corpus, and return the tf-idf vector from sklearn
    """
    tf = TfidfVectorizer(max_features=100000, stop_words='english', max_df=0.9,
                         tokenizer=tokenizing, encoding='cp1252')
    tf.fit(corpus)
    
    return tf


def tfidf_sentences(tfidf_vector, text, stemming=True):
    """
    creates a list of sentences and tranforms them using the tf-idf vector
    returns the list of sentences and the tranformed vectores of the sentences
    """

    #getting the sentences:
    sentences = sent_tokenize(text)
    
    #if we are working with stemmed corpus we also need to stemm the words from 
    #the text we want to summarize before transforming it with tfidf
    # if not we can just transform the text without any other pre processing
    if stemming:
        stemmed_sentences = []
        tokenized_sentences = [word_tokenize(sent) for sent in sentences]
        for sentence in tokenized_sentences:
            stemmed_sentences.append(' '.join([stemmer.stem(word) for word in sentence]))
        
        text_vect = tfidf_vector.transform(stemmed_sentences)
    else:   
        text_vect = tfidf_vector.transform(sentences)
    
    return sentences, text_vect


def tfidf_summarizer(tfidf_vector, text, number_of_sentences, stemming=True, 
                     postprocessing=True, post_sentences=3, post_factor=1.1):
    """
    Calls the function tfidf_sentences to get the sentences and the transformed vectores
    calculates the total importance of each sentence and divides by the number of words
    Sorts and prints the most important sentences by the order they appear in the text
    """
    
    sentences, text_vect = tfidf_sentences(tfidf_vector, text, stemming=stemming)
    
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
            
    if postprocessing:
        sentence_importance = post_processing(sentence_importance, n_sentences=post_sentences, 
                                              factor=post_factor )
        
    #sorting the importance of the sentences in descending order
    order = np.array(sentence_importance).argsort()[::-1]
     
    #printing the sentences following the order in the text
    for i in order[:number_of_sentences]:
        print (sentences[i])
        

"""
here starts text rank summarizer, based on graphs and PageRank
"""

def create_graph(tfidf_vector, text, stemming=True):
    """
    Creates a graph for the text with the sentences as nodes and the cosine similarity 
    between the sentences as the edges
    returns the graph and a list with the sentences
    """
    #initialize networkx graph
    graph = nx.Graph()
    
    #getting the sentences and the tfidf vectores od the sentences
    sentences, text_vect = tfidf_sentences(tfidf_vector, text, stemming=stemming)
    
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
        


def textrank_summarizer(tfidf_vector, text, number_of_sentences, 
                        postprocessing=True, post_sentences=3, post_factor=1.1):
    """
    Calls the function create_graph to create the graph and calculates the Pagerank value
    of the nodes. Sorts the sentences in descending order of importance
    Prints the most important sentences, depending of the number_sentences
    
    """
    #creating the graph
    graph, sentences = create_graph(tfidf_vector, text)
    
    #calculate Pagerank
    rank = nx.pagerank(graph, weight='weight')
    
    #convert the rank of sentences to an array
    sentence_importance = []
    for v in rank.values():
        sentence_importance.append(v[0][0])
        
    if postprocessing:
        sentence_importance = post_processing(sentence_importance, n_sentences=post_sentences, 
                                              factor=post_factor )
    
    #sorting by Rank value
    order = np.array(sentence_importance).argsort()[::-1]
    
    #Printing the sentences with highest rank
    for i in order[:number_of_sentences]:
        print (sentences[i])   


def post_processing(sentence_importance, n_sentences=3, factor=1.1):
    """
    The introduction and conclusion of a text usually are more important.
    using the weight list of the sentences will increase the importance of number_sentences
    in the beginning and in the end of the text by the importance_factor
    """
    sentence_importance2 = sentence_importance[:]
    for i in range(n_sentences):
        sentence_importance2[i] *= factor
        sentence_importance2[i*-1] *= factor
        
    return sentence_importance2


def get_key_words(tfidf_vector, text, n_words=10):
    """
    prints the most import important n_words from the text    
    """
    #stemming and tranforming the text first
    text_stemmed = ' '.join(tokenizing(text))
    
    vector = tfidf_vector.transform([text_stemmed])
    
    print('The most important words in this document are: ')
    
    #gets the important stemmed words and find those words in the text to print the original
    for index in vector.toarray()[0].argsort()[::-1][:n_words]:
        stemmed_word = tfidf_vector.get_feature_names()[index]
        
        indices = re.search('{}\S*'.format(str(stemmed_word)), text.lower()).span()
        print('  -{}'.format(text[indices[0]:indices[1]]))
        
    

#df = pd.read_csv(r'C:\bts_master\project\news\data.csv', encoding='cp1252')
#corpus = df.text.values

#CREATE THE TF-IDF VECTOR BASED ON A CORPUS:
#tfidf_vector = build_tfidf(corpus)

#To avoid creating a tfidf vectorizer every time, it takes time to do it:
#save the tfidf vectorizer to a file
#pickle.dump(tfidf_vector, open(r"C:\Projects\Projects\summarizer\tfidf_vector.pickle", "wb"))

#load the tfidf vectorizer
tfidf_vector = pickle.load(open("tfidf_vector.pickle", "rb"))
                    
#SUMMARIZE, text is the text to summarize and the number of sentences is how many sentences you want
#textrank_summarizer(tfidf_vector, text, number_sentences)
#or
#tfidf_summarizer(tfidf_vector, text, number_of_sentences)
