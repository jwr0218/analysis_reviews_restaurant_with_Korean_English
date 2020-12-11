
# coding: utf-8

# Scraping TripAdvisor Reviews

# In[3]:


#import the libraries as needed
import requests
from selenium import webdriver
from bs4 import BeautifulSoup
import time
import csv
import collections
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
import string
from nltk.tokenize import word_tokenize


# In[4]:


#using Chromedriver to open webpages
browser = webdriver.Chrome('chromedriver.exe')
#Headers will make it look like you are using a web browser
headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.90 Safari/537.36'}
#We will use the iteration to retrieve and scrape the web pages, reviews, and ratings from each page on Trip Advisor
reviews = []
ratelist = []
for i in range(0,455,5):
    url = 'https://www.tripadvisor.com/Hotel_Review-g60971-d115318-Reviews-or{}-Hotel_Monte_Vista-Flagstaff_Arizona.html'.format(i)
    browser.get(url)
    time.sleep(5)
    element_list = browser.find_elements_by_xpath("//span[@class='taLnk ulBlueLinks']")
    #Iteration clicks all of the 'More' links. The 'try' statement allows the iteration 
    #to continue with 'pass' when an error message appears-caused by TA.
    for e in element_list:
        try:
            e.click()
        except:
            pass
        #Variable to get the page source through BeautifulSoup.
    html = browser.page_source
    response = requests.get(url, headers=headers, verify=False).text
    soup = BeautifulSoup(response, "lxml")
#Looping through 'div' 'reviewSelector' will help find all the review containers we need in each page that have rating and review
    for r in soup.find_all('div', 'reviewSelector'):
        review = r.p.text
#Cleaning the lemmas or words in reviews now will make it easier when we start predictive modeling
        words = word_tokenize(review)
        words = word_tokenize(review.replace('\n',' '))
        clean_words = [word.lower() for word in words if word not in set(string.punctuation)]
        characters_to_remove = ["''",'``','...']
        clean_words = [word for word in clean_words if word not in set(characters_to_remove)]
        english_stops = set(stopwords.words('english'))
        clean_words = [word for word in clean_words if word not in english_stops]
        wordnet_lemmatizer = WordNetLemmatizer()
        lemma_list = [wordnet_lemmatizer.lemmatize(word) for word in clean_words]
        reviews.append(lemma_list)
#Here we are using a simple control flow to recode the ratings for our model. If rating is 1-3 negative, else positive
        rating = int(r.find('span','ui_bubble_rating')['class'][1].split('_')[1])/10
        if rating == 1 or rating == 2 or rating == 3:
            ratelist.append('negative')
        elif rating == 4 or rating == 5:
            ratelist.append('positive')

browser.quit()       
print("Finished!")


# In[6]:


#looking at both length of reviews and ratings
print(len(reviews))
print(len(ratelist))


# In[8]:


#Making a variable that zips together reviews and ratings for modeling
rl = zip(reviews,ratelist)


# In[9]:


# Define a function that receives a list of words and returns a dictionary 
def bag_of_words(words):
    return dict([(word, True) for word in words])


# In[10]:


# Define another function that will return words that are in words, but not in badwords
def bag_of_words_not_in_set(words, badwords):
    return bag_of_words(set(words) - set(badwords))


# In[11]:


# Define another function that will return words that are in words, but not in stopwords
from nltk.corpus import stopwords

def bag_of_non_stopwords(words, stopfile='english'):
    badwords = stopwords.words(stopfile)
    return bag_of_words_not_in_set(words, badwords)


# In[12]:


#Make training data for modeling
train_data = []

for r, v in rl:
    bag_of_words(r)
    train_data.append((bag_of_words(r),v))


# In[13]:


#Random shuffling the training data for modeling
import random
random.shuffle(train_data)
print(len(train_data))


# In[14]:


#Spliting training and test data
train_set, test_set = train_data[0:340], train_data[340:]


# In[15]:


#find the most informative features using Naive Bayes Classifier
import nltk
import collections
from nltk.metrics.scores import (accuracy, precision, recall, f_measure)
classifier = nltk.NaiveBayesClassifier.train(train_set)
classifier.show_most_informative_features(10)


# In[16]:


print(nltk.classify.accuracy(classifier, test_set))


# What are the key words (lemmas) that predict the rating? (5 pts.)
# 
# The key words (lemmas) that predict the ratings are the top 5: building, corner, well, spacious, and loved.
