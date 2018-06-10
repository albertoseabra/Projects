#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 22:33:57 2018

@author: alberto
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
import requests
from bs4 import BeautifulSoup

stemmer = LancasterStemmer()


class Summarizer:
    
    def __init__(self, tfidf_vector, text=None, url=None, stemming=True):
        self.tfidf_vector = tfidf_vector
        self.title = None
        self.url = url
        
        if text is not None:
            self.text = text
            self.paragraphs = [line for line in self.text.split('\n') if len(line) > 10]
            
        elif url is not None:
            self.paragraphs, self.title = self.scrape_website()
            self.text = '\n'.join(self.paragraphs) 
            
        else:
            raise Exception('You need a text or a url to summarize')

        self.sentences = self.get_sentences()
        self.sentences_vect = self.tfidf_text_list(self.sentences, stemming=stemming)
        self.paragraph_vect = self.tfidf_text_list(self.paragraphs, stemming=stemming)
        self.sentence_weights_graph = []
        self.sentence_weights_tfidf = []
        self.paragraph_weights_graph = []
        self.paragraph_weights_tfidf = []
                
        print("this text has {} sentences and {} paragraphs, you can summarize using one or the other."
              .format(len(self.sentences), len(self.paragraphs)))

    def scrape_website(self):
        """
        Extracts the title and a list with the paragraphs of the text
        """
        page = requests.get(self.url)
        
        if page.status_code == requests.codes.ok:
            soup = BeautifulSoup(page.content, 'lxml')
            
            # find the text
            text = [line.text for line in soup.findAll('p')]
            
            paragraphs = [line for line in text if len(line) > 20]

            try:
                title = soup.find('h1').text.strip()

                return paragraphs, title
            except:
                print('Failed to extract title')

            return paragraphs
        
        else:
            print('Something went wrong trying to access the URL')

    def get_sentences(self):
        """
        transforms the paragraphs in list of sentnences
        """
        sentences_list = [sent_tokenize(paragr) for paragr in self.paragraphs]
        # flatten the list of lists
        sentences = [sentence for sublist in sentences_list for sentence in sublist]
        # filter potential short sentences that dont add much to the story
        # or are thing like "share this", "advertising", things that shouldn't be part
        sentences = [sentence for sentence in sentences if len(sentences) > 20]
        
        return sentences
        
    def tfidf_text_list(self, text_list, stemming):
        """
        transforms the text, list of sentences or list of paragraphs,
        in tfidf vectors and returns them
        """
        # if we are working with stemmed corpus we also need to stem the words from
        # the text we want to summarize before transforming it with tfidf
        # if not we can just transform the text without any other pre processing
        if stemming:
            stemmed_sentences = []
            tokenized_sentences = [word_tokenize(sent) for sent in text_list]
            for sentence in tokenized_sentences:
                stemmed_sentences.append(' '.join([stemmer.stem(word) for word in sentence]))
            
            text_vect = self.tfidf_vector.transform(stemmed_sentences)
        else:   
            text_vect = self.tfidf_vector.transform(text_list)
        
        return text_vect

    def key_words(self, n_words=10):
        """
        prints the most import important n_words from the text    
        """
        # stemming and transforming the text first
        tokens = [word for sent in sent_tokenize(self.text) for word in word_tokenize(sent)]
        stemmed = [stemmer.stem(word) for word in tokens]
        text_stemmed = ' '.join(stemmed)
        
        vector = self.tfidf_vector.transform([text_stemmed])
        
        print('The top Words are: ')
        # gets the important stemmed words and find those words in the text to print the original
        for index in vector.toarray()[0].argsort()[::-1][:n_words]:
            stemmed_word = self.tfidf_vector.get_feature_names()[index]
            
            indices = re.search('{}\S*'.format(str(stemmed_word)), self.text.lower()).span()
            print(' {};'.format(self.text[indices[0]:indices[1]]), end=' ')
        print()

    def tfidf_summary(self, number_of_sentences, sentences=True,
                      postprocessing=False, post_sentences=1, post_factor=1.1):
        """
        calculates the total importance of each sentence or paragraph and divides by the number of words
        Sorts and prints the most important sentences/paragraphs by the order they appear in the text
        Will use paragrphs as default, to use sentences need to use sentences=True
        """
        # if we are working with sentences:
        if sentences:
            # better to start with a new weights list everytime the summary is called 
            self.sentence_weights_tfidf = []
            # iterate for each vector, get the sum of tf-idf and then divide by the number of words
            for i in range(self.sentences_vect.shape[0]):
                soma = self.sentences_vect[i].sum()
                words = self.sentences_vect[i].count_nonzero()
                # in case a sentence has only stopwords and/or numbers the sum of tf-idf will be zero
                # we give it the importance value of 0
                if soma == 0:
                    self.sentence_weights_tfidf.append(0)
                # in case is a really short sentence it might get high importance value
                # but it wont give us much information, going to ignore them
                elif words < 5:
                    self.sentence_weights_tfidf.append(0)
                else:
                    self.sentence_weights_tfidf.append(soma/words)
                    
            if postprocessing:
                self.sentence_weights_tfidf = self.post_processing(self.sentence_weights_tfidf, 
                                                                   post_sentences, post_factor)
                
            # sorting the importance of the sentences in descending order
            order = np.array(self.sentence_weights_tfidf).argsort()[::-1]
            
        else:
            self.paragraph_weights_tfidf = []
            
            for i in range(self.paragraph_vect.shape[0]):
                soma = self.paragraph_vect[i].sum()
                words = self.paragraph_vect[i].count_nonzero()
                if soma == 0:
                    self.paragraph_weights_tfidf.append(0)
                elif words < 5:
                    self.paragraph_weights_tfidf.append(0)
                else:
                    self.paragraph_weights_tfidf.append(soma/words)
                    
            if postprocessing:
                self.paragraph_weights_tfidf = self.post_processing(self.paragraph_weights_tfidf, 
                                                                    post_sentences, post_factor)
                
            # sorting the importance of the sentences in descending order
            order = np.array(self.paragraph_weights_tfidf).argsort()[::-1]            
            
        self.print_summary(order, number_of_sentences, sentences)

    def post_processing(self, sentence_weights, n_sentences, factor):
        """
        The introduction and conclusion of a text usually are more important.
        using the weight list of the sentences will increase the importance of number_sentences
        in the beginning and in the end of the text by the importance_factor
        """
        sentence_weights2 = sentence_weights[:]
        for i in range(n_sentences):
            sentence_weights2[i] *= factor
            sentence_weights2[i*-1] *= factor
            
        return sentence_weights2

    def create_graph(self, text_list, text_vector):
        """
        Creates a graph for the text with the sentences/paragraphs as nodes and 
        the cosine similarity between the sentences as the edges
        returns the graph
        """
        # initialize networkx graph
        graph = nx.Graph()
        
        # iterates of each pair of sentences
        size = len(text_list)
        for index1 in range(size):
            for index2 in range(index1 + 1, size):
                if index1 == index2:
                    continue
                # calculates the similarity between the pair of sentences
                # creates and graph edge between the sentences with the similarity as edge weight
                else:
                    graph.add_edge(index1, index2, weight=cosine_similarity(text_vector[index1].toarray(),
                                                                            text_vector[index2].toarray()))
        
        return graph
    
    def textrank_summary(self, number_of_sentences, sentences=True,
                         postprocessing=False, post_sentences=1, post_factor=1.1):
        """
        Calls the function create_graph to create the graph and calculates the Pagerank value
        of the nodes. Sorts the sentences in descending order of importance
        Prints the most important sentences, depending of the number_sentences
        
        """
        if sentences:
            # better to start with a new weights list everytime the summary is called 
            self.sentence_weights_graph = []
            # creating the graph
            graph = self.create_graph(self.sentences, self.sentences_vect)
            # calculate Pagerank
            rank = nx.pagerank(graph, weight='weight')
            
            # convert the rank of sentences to an array
            for v in rank.values():
                self.sentence_weights_graph.append(v[0][0])
                
            if postprocessing:
                self.sentence_weights_graph = self.post_processing(self.sentence_weights_graph, 
                                                                   post_sentences, post_factor)
            
            # sorting by Rank value
            order = np.array(self.sentence_weights_graph).argsort()[::-1]
        
        else:
            self.paragraph_weights_tfidf = [] 
            graph = self.create_graph(self.paragraphs, self.paragraph_vect)
            rank = nx.pagerank(graph, weight='weight')
            
            for v in rank.values():
                self.paragraph_weights_graph.append(v[0][0])
                
            if postprocessing:
                self.paragraph_weights_graph = self.post_processing(self.paragraph_weights_graph, 
                                                                   post_sentences, post_factor)
            
            # sorting by Rank value
            order = np.array(self.paragraph_weights_graph).argsort()[::-1]        
        
        self.print_summary(order, number_of_sentences, sentences)


    def print_summary(self, ordered_list, number_of_sentences, sentences):
        """
        just to print the summary giving the list of indices ordered by importance
        """
        if self.title is not None:
            print("Title: ", self.title)
        print()    
        self.key_words(5)
        print()
        
        # printing the sentences following the order in the text
        indices = sorted(ordered_list[:number_of_sentences])
        for i in indices:
            if sentences:
                print (self.sentences[i])
            else:
                print (self.paragraphs[i])        
        
            
# load the tfidf vectorizer
# tfidf_vector = pickle.load(open("tfidf_vector250k.pickle", "rb"))
#
# data = pd.read_csv('/home/alberto/Downloads/news_summary.csv', encoding='cp437')
         

# TO BUILD A NEW TFIDF VECTOR:
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
    tf = TfidfVectorizer(max_features=250000, stop_words='english', max_df=0.8,
                         tokenizer=tokenizing, encoding='cp1252')
    tf.fit(corpus)
    
    return tf

# CREATE THE TF-IDF VECTOR BASED ON A CORPUS:
# tfidf_vector = build_tfidf(corpus)