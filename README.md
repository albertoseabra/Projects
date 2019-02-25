# Most recent project:
[An "inpirational" quote generator with a twitter bot script to post the quotes](https://github.com/albertoseabra/quotes_generator_twitter)

# Exploratory Data Analysis, Data Cleaning, Feature Engineering & Machine Learning
#### https://github.com/albertoseabra/Projects/blob/master/exploration/adults_analysis.ipynb
EDA of a data set (https://archive.ics.uci.edu/ml/datasets/Adult) with census data of the USA. Includes information like age, 
ocupation, education, hours worked. The target is if the person makes more or less than 50k. 

#### https://github.com/albertoseabra/Projects/blob/master/exploration/bike_sharing_analysis.ipynb
EDA of a dataset of the number of bike rentals with information like date and time, weather, temperature, workday.

#### https://github.com/albertoseabra/Projects/blob/master/exploration/flights.ipynb
EDA and Feature engineering of a dataset of the flights in the USA during the year of 2015. The objective is to analyse what influences
the delays of the flights. Plenty of variables available.
Final objective is to predict if a flight will arrive delayed or not, predictions in this file:
https://github.com/albertoseabra/Projects/blob/master/exploration/flights_predictions.ipynb

#### https://github.com/albertoseabra/Projects/blob/master/final_project/news_separated.ipynb
EDA and cleaning of a dataset of different types of news. To be used in the future.

#### https://github.com/albertoseabra/Projects/blob/master/ML_mammogram.ipynb
Trying to predict if a mammogram mass is benign or not with several different ML algorithms.

#### https://github.com/albertoseabra/Projects/blob/master/machine_learning/cajamar.ipynb
Predicting the next product each customer will subscribe between 94 different possible products.
Predictions based on products already subscribed and the date of subsciption, and social demographic information.
Feature Engineering

## Text Summarizer
### New Version: https://github.com/albertoseabra/Projects/blob/master/summarizer/summarizerembeddings.py
New version with using word embeddings instead of TF-IDF vectors   
Can make summaries also with different methods:
+ Implementing Textrank but now first the sentences are transformed using word embeddings and then is calculated the similarity between them
+ A summarizer that just selects the sentences that are most similar with the full text, ends up givin the same results as Textrank
+ A greedy algorith that adds to the summary the sentence that gives the largest increase in similarity between the full text and the summary, avoiding that way the problem of adding redundant sentences, sentences that are similar to the full text but don't increase the ammount of information in the summary

#### 1st version now with OOP https://github.com/albertoseabra/Projects/blob/master/summarizer/summaryOOP.py

Extracts a summary of a text. Using two different methods:  
  - Selecting most important sentences based on tf-idf    
  - Creating a graph, calculating similarity between sentences and applying PageRank to select the most important sentences   
Also extracts the most important words from a text, works with a url or with the text of the news, and can do the summary with sentences or paragraphs.   
older version: https://github.com/albertoseabra/Projects/tree/master/summarizer


# Others
#### https://github.com/albertoseabra/Projects/blob/master/beer_recipe_webscraper.ipynb
Web scrapping of all the recipes of beers from a website with complete information


#### https://github.com/albertoseabra/Projects/tree/master/blackjack
BlackJack game with multiplayer. With OOP.
