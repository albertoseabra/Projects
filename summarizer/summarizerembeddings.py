import numpy as np
import networkx as nx
import requests
from bs4 import BeautifulSoup
import spacy
nlp = spacy.load('en_core_web_lg')
#glove embeddings:
#nlp = spacy.load('en_vectors_web_lg')

class SummarizerEmbeddings:

    def __init__(self, text=None, url=None):
        self.title = None
        self.url = url
        self.sentence_weights = []

        if text is not None:
            self.text = text
            self.paragraphs = [line for line in self.text.split('\n') if len(line) > 20]

        elif url is not None:
            self.paragraphs, self.title = self.scrape_website()
            self.text = '\n'.join(self.paragraphs)

        else:
            raise Exception('You need a text or a url to summarize')

        self.embedded_text = nlp(self.text)
        sentences = list(self.embedded_text.sents)
        self.sentences = [sentence for sentence in sentences if len(sentence) > 20]

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

    def create_graph(self):
        """
        Creates a graph for the text with the sentences/paragraphs as nodes and
        the similarity between the sentences as the edges
        returns the graph
        """
        # initialize networkx graph
        graph = nx.Graph()

        # iterates of each pair of sentences
        size = len(self.sentences)
        for index1 in range(size):
            for index2 in range(index1 + 1, size):
                if index1 == index2:
                    continue
                # calculates the similarity between the pair of sentences
                # creates and graph edge between the sentences with the similarity as edge weight
                else:
                    graph.add_edge(index1, index2, weight=(self.sentences[index1]).similarity(self.sentences[index2]))

        return graph

    def textrank_summary(self, number_of_sentences,):
        """
        Calls the function create_graph to create the graph and calculates the Pagerank value
        of the nodes. Sorts the sentences in descending order of importance
        Prints the most important sentences, depending of the number_sentences

        """
        # better to start with a new weights list every time the summary is called
        self.sentence_weights = []
        # creating the graph
        graph = self.create_graph()
        # calculate Pagerank
        rank = nx.pagerank(graph, weight='weight')

        # convert the rank of sentences to an array
        for v in rank.values():
            self.sentence_weights.append(v)

        # sorting by Rank value
        order = np.array(self.sentence_weights).argsort()[::-1]

        self.print_summary(order, number_of_sentences)


    def print_summary(self, ordered_list, number_of_sentences):
        """
        just to print the summary giving the list of indices ordered by importance
        """
        if self.title is not None:
            print("Title: ", self.title)
        print()

        # printing the sentences following the order in the text
        indices = sorted(ordered_list[:number_of_sentences])
        for i in indices:
                print(self.sentences[i])



    def print_similarities(self):

        text_embedding = nlp(self.text)

        print("similarity between the full text and title: {}".format(text_embedding.similarity(nlp(self.title))))

        for i in range(len(self.sentences)):
            print("similarity between full text and sentences {} is {}".format(i, text_embedding.similarity(self.sentences[i])))











