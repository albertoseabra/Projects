import numpy as np
import networkx as nx
import requests
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import spacy
nlp = spacy.load('en_core_web_lg')
#glove embeddings:
#nlp = spacy.load('en_vectors_web_lg')


class SummarizerEmbeddings:
    """

    """

    def __init__(self, text=None, url=None):
        self.title = None
        self.url = url
        self.sentence_weights = []

        if text is not None:
            self.text = text
            self.paragraphs = [line for line in self.text.split('\n') if len(line) > 10]

        elif url is not None:
            self.paragraphs, self.title = self.scrape_website()
            self.text = '\n'.join(self.paragraphs)

        else:
            raise Exception('You need a text or a url to summarize')

        self.embedded_text = nlp(self.text)
        sentences = list(self.embedded_text.sents)
        self.sentences = [sentence for sentence in sentences if len(sentence) > 10]

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
        :param ordered_list:
        :param number_of_sentences:
        :return:
        """
        if self.title is not None:
            print("Title: ", self.title)
        print()

        # printing the sentences following the order in the text
        indices = sorted(ordered_list[:number_of_sentences])
        for i in indices:
                print(self.sentences[i])


    def docembedding_summary(self, number_of_sentences):
        """
        Creates a summary selecting the sentences that are more similar to the full text
        first creates the embedding of the full text and then calculates the similarity with
        each of the sentences of the text
        :param number_of_sentences: how many sentences the summary should have
        :return: no return, just prints the summary made
        """

        sentences_simil = []
        for i in range(len(self.sentences)):
            sim = self.embedded_text.similarity(self.sentences[i])
            sentences_simil.append(sim)

        order = np.array(sentences_simil).argsort()[::-1]

        self.print_summary(order, number_of_sentences)


    def greedy_summary(self, number_of_sentences):
        """
        greedy algorithm, selects the sentences, one by one, that create the largest increase in the similarity
        between the full text and the summary
        :param number_of_sentences: how many sentences the summary should contain
        :return:
        """
        summary_indices = []
        summary = ""
        sentences_simil = []
        # getting the most similar sentence first:
        for i in range(len(self.sentences)):
            sim = self.embedded_text.similarity(self.sentences[i])
            sentences_simil.append(sim)

        # add it to the summary and to a list of indices of sentences
        summary += str(self.sentences[np.array(sentences_simil).argmax()])
        summary_indices.append(np.array(sentences_simil).argmax())

        n_sentences = number_of_sentences
        while n_sentences > 1:
            sentence_to_add = []
            # for each sentence add it to the summary and test the similarity between the new summary and the full text
            # choose the sentence that gives a higher final similarity between text and summary
            for i, sentence in enumerate(self.sentences):
                summary_test = summary + str(sentence)
                similarity_test = self.embedded_text.similarity(nlp(summary_test))

                if len(sentence_to_add) == 0:
                    sentence_to_add.append((i, similarity_test))
                elif similarity_test > sentence_to_add[0][1]:
                    sentence_to_add[0] = (i, similarity_test)
                else:
                    continue
            summary += str(self.sentences[sentence_to_add[0][0]])
            summary_indices.append(sentence_to_add[0][0])
            n_sentences -= 1

        order = sorted(summary_indices)

        self.print_summary(order, number_of_sentences)
        print('Similarity between the full text and the summary: {}'.format(self.embedded_text.similarity(nlp(summary))))


    def centroid_summary(self, number_of_sentences):
        """
        Divides de document in number_of_sentences clusters using K-Means
        from each cluster selects the sentences more similar to the cluster center to create a summary
        :param number_of_sentences:
        :return:
        """


        sentences_array = np.array([sentence.vector for sentence in self.sentences])

        cluster = KMeans(number_of_sentences).fit(sentences_array)

        closest_sentences = []
        for i, center in enumerate(cluster.cluster_centers_):
            closest = [(0,0)]
            for j, sentence in enumerate(self.sentences):
                if cluster.labels_[j] == i:
                    sim = cosine_similarity(cluster.cluster_centers_[i].reshape(1,-1), self.sentences[j].vector.reshape(1,-1))
                    if sim > closest[0][1]:
                        closest = [(j,sim)]

            closest_sentences.append(closest)

        summary_indices = [l[0][0] for l in closest_sentences]

        self.print_summary(summary_indices, number_of_sentences)


    def centroid_summary2(self, number_of_sentences):
        """
        a bit more complex than the previous: first divides document using K-Means
        for each cluster compares the similarity of the full text of the cluster with each sentence that is part of
        that cluster and selects the sentence with the highest similarity with the full text of the custer
        :param number_of_sentences:
        :return:
        """

        sentences_array = np.array([sentence.vector for sentence in self.sentences])

        cluster = KMeans(number_of_sentences).fit(sentences_array)

        final_summary = []
        # for each cluster creates a text with the sentences that belong to that cluster
        for i, center in enumerate(cluster.cluster_centers_):
            text = ""
            sentences_indices = []
            for j, sentence in enumerate(self.sentences):
                if cluster.labels_[j] == i:
                    text += str(self.sentences[j])
                    sentences_indices.append(j)

            # compares the similarity of each sentence with the full text of the cluster
            sentences_simil = []
            for indice in sentences_indices:
                sim = self.sentences[indice].similarity(nlp(text))
                sentences_simil.append(sim)
            # append to the summary the sentence with the highest similarity
            final_summary.append(sentences_indices[np.argmax(sentences_simil)])

        self.print_summary(final_summary, number_of_sentences)




    def print_similarities(self):

        text_embedding = nlp(self.text)

        print("similarity between the full text and title: {}".format(text_embedding.similarity(nlp(self.title))))

        sentences_simil = []
        for i in range(len(self.sentences)):
            sim = text_embedding.similarity(self.sentences[i])
            sentences_simil.append(sim)
            print("similarity between full text and sentences {} is {}".format(i, sim))

        return np.array(sentences_simil).argsort()[::-1]
