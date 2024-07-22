# This is the 12th article in my series of articles on Python for NLP. In the previous article, I briefly explained the different functionalities of the Python's Gensim library. Until now, in this series, we have covered almost all of the most commonly used NLP libraries such as NLTK, SpaCy, Gensim, StanfordCoreNLP, Pattern, TextBlob, etc.

# In this article, we are not going to explore any NLP library. Rather, we will develop a very simple rule-based chatbot capable of answering user queries regarding the sport of Tennis. But before we begin actual coding, let's first briefly discuss what chatbots are and how they are used.

# What is a Chatbot?
# A chatbot is a conversational agent capable of answering user queries in the form of text, speech, or via a graphical user interface. In simple words, a chatbot is a software application that can chat with a user on any topic. Chatbots can be broadly categorized into two types: Task-Oriented Chatbots and General Purpose Chatbots.

# The task-oriented chatbots are designed to perform specific tasks. For instance, a task-oriented chatbot can answer queries related to train reservation, pizza delivery; it can also work as a personal medical therapist or personal assistant.

# On the other hand, general purpose chatbots can have open-ended discussions with the users.

# There is also a third type of chatbots called hybrid chatbots that can engage in both task-oriented and open-ended discussion with the users.

# Approaches for Chatbot Development
# Chatbot development approaches fall in two categories: rule-based chatbots and learning-based chatbots.

# Learning-Based Chatbots
# Learning-based chatbots are the type of chatbots that use machine learning techniques and a dataset to learn to generate a response to user queries. Learning-based chatbots can be further divided into two categories: retrieval-based chatbots and generative chatbots.


# The retrieval based chatbots learn to select a certain response to user queries. On the other hand, generative chatbots learn to generate a response on the fly.

# One of the main advantages of learning-based chatbots is their flexibility to answer a variety of user queries. Though the response might not always be correct, learning-based chatbots are capable of answering any type of user query. One of the major drawbacks of these chatbots is that they may need a huge amount of time and data to train.

# Like rule-based models, retrieval-based models rely on predefined responses, but they have the additional ability to self-learn and improve their selection of response over time. Finally, generative chatbots are capable of formulating their own original responses based on user input, rather than relying on existing text.

# Rule-Based Chatbots
# Rule-based chatbots are pretty straight forward as compared to learning-based chatbots. There are a specific set of rules. If the user query matches any rule, the answer to the query is generated, otherwise the user is notified that the answer to user query doesn't exist.

# One of the advantages of rule-based chatbots is that they always give accurate results. However, on the downside, they do not scale well. To add more responses, you have to define new rules.

# In the following section, I will explain how to create a rule-based chatbot that will reply to simple user queries regarding the sport of tennis.

# Rule-Based Chatbot Development with Python
# The chatbot we are going to develop will be very simple. First we need a corpus that contains lots of information about the sport of tennis. We will develop such a corpus by scraping the Wikipedia article on tennis. Next, we will perform some preprocessing on the corpus and then will divide the corpus into sentences.

# Corpus: a collection of written texts, especially the entire works of a particular author or a body of writing on a particular subject

# When a user enters a query, the query will be converted into vectorized form. All the sentences in the corpus will also be converted into their corresponding vectorized forms. Next, the sentence with the highest cosine similarity with the user input vector will be selected as a response to the user input.

import nltk
import numpy as np
import random
import string
import bs4 as bs
import urllib.request
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Creating the Corpus
# As we said earlier, we will use the Wikipedia article on Tennis to create our corpus. The following script retrieves the Wikipedia article and extracts all the paragraphs from the article text. Finally the text is converted into the lower case for easier processing.


# URL = "http://www.values.com/inspirational-quotes" 
# r = requests.get(URL) 
  
# soup = BeautifulSoup(r.content, 'html5lib') # If this line causes an error, run 'pip install html5lib' or install html5lib 
# print(soup.prettify()) <-- the method I'm familiar with

raw_html = urllib.request.urlopen('https://en.wikipedia.org/wiki/Tennis')
raw_html = raw_html.read()

article_html = bs.BeautifulSoup(raw_html, 'html.parser')

article_paragraphs = article_html.find_all('p')

article_text = ''

for para in article_paragraphs:
    article_text += para.text

article_text = article_text.lower()


# Text Preprocessing and Helper Function
# Next, we need to preprocess our text to remove all the special characters and empty spaces from our text. The following regular expression does that:

article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)
article_text = re.sub(r'\s+', ' ', article_text)


# We need to divide our text into sentences and words since the cosine similarity of the user input will actually be compared with each sentence. Execute the following script:

article_sentences = nltk.sent_tokenize(article_text)
article_words = nltk.word_tokenize(article_text)

# Finally, we need to create helper functions that will remove the punctuation from the user input text and will also lemmatize the text. Lemmatization refers to reducing a word to its root form. For instance, lemmatization the word "ate" returns eat, the word "throwing" will become throw and the word "worse" will be reduced to "bad". Execute the following code:

wnlemmatizer = nltk.stem.WordNetLemmatizer()

def perform_lemmatization(tokens):
    return [wnlemmatizer.lemmatize(token) for token in tokens]

punctuation_removal = dict((ord(punctuation), None) for punctuation in string.punctuation)

def get_processed_text(document):
    return perform_lemmatization(nltk.word_tokenize(document.lower().translate(punctuation_removal)))

# In the script above we first instantiate the WordNetLemmatizer from the NTLK library. Next, we define a function perform_lemmatization, which takes a list of words as input and lemmatize the corresponding lemmatized list of words. The punctuation_removal list removes the punctuation from the passed text. Finally, the get_processed_text method takes a sentence as input, tokenizes it, lemmatizes it, and then removes the punctuation from the sentence.

# Responding to Greetings
# Since we are developing a rule-based chatbot, we need to handle different types of user inputs in a different manner. For instance, for greetings we will define a dedicated function. To handle greetings, we will create two lists: greeting_inputs and greeting_outputs. When a user enters a greeting, we will try to search it in the greetings_inputs list, if the greeting is found, we will randomly choose a response from the greeting_outputs list.

# Look at the following script:

greeting_inputs = ("hey", "good morning", "good evening", "morning", "evening", "hi", "whatsup")
greeting_responses = ["hey", "hey hows you?", "*nods*", "hello, how you doing", "hello", "Welcome, I am good and you"]

def generate_greeting_response(greeting):
    for token in greeting.split():
        if token.lower() in greeting_inputs:
            return random.choice(greeting_responses)
        

# Responding to User Queries
# As we said earlier, the response will be generated based upon the cosine similarity of the vectorized form of the input sentence and the sentences in the corpora. The following script imports the TfidfVectorizer and the cosine_similarity functions: above ^^


def generate_response(user_input):
    tennisrobo_response = ''
    article_sentences.append(user_input)

    # You can see that the generate_response() method accepts one parameter which is user input. Next, we define an empty string tennisrobo_response. We then append the user input to the list of already existing sentences. After that in the following lines:

    word_vectorizer = TfidfVectorizer(tokenizer=get_processed_text, stop_words='english')
    all_word_vectors = word_vectorizer.fit_transform(article_sentences)
    # We initialize the tfidfvectorizer and then convert all the sentences in the corpus along with the input sentence into their corresponding vectorized form.
    # In the following line:

    similar_vector_values = cosine_similarity(all_word_vectors[-1], all_word_vectors)

    # We use the cosine_similarity function to find the cosine similarity between the last item in the all_word_vectors list (which is actually the word vector for the user input since it was appended at the end) and the word vectors for all the sentences in the corpus.

    # Next, in the following line:
    similar_sentence_number = similar_vector_values.argsort()[0][-2]

    # We sort the list containing the cosine similarities of the vectors, the second last item in the list will actually have the highest cosine (after sorting) with the user input. The last item is the user input itself, therefore we did not select that.

    matched_vector = similar_vector_values.flatten()
    matched_vector.sort()
    vector_matched = matched_vector[-2]

    if vector_matched == 0:
        tennisrobo_response = tennisrobo_response + "I am sorry, I could not understand you"
        return tennisrobo_response
    else:
        tennisrobo_response = tennisrobo_response + article_sentences[similar_sentence_number]
        return tennisrobo_response
    # Finally, we flatten the retrieved cosine similarity and check if the similarity is equal to zero or not. If the cosine similarity of the matched vector is 0, that means our query did not have an answer. In that case, we will simply print that we do not understand the user query.

    # Otherwise, if the cosine similarity is not equal to zero, that means we found a sentence similar to the input in our corpus. In that case, we will just pass the index of the matched sentence to our "article_sentences" list that contains the collection of all sentences.

continue_dialogue = True
print("Hello, I am your friend TennisRobo. You can ask me any question regarding tennis:")
while(continue_dialogue == True):
    human_text = input()
    human_text = human_text.lower()
    if human_text != 'bye':
        if human_text == 'thanks' or human_text == 'thank you very much' or human_text == 'thank you':
            continue_dialogue = False
            print("TennisRobo: Most welcome")
        else:
            if generate_greeting_response(human_text) != None:
                print("TennisRobo: " + generate_greeting_response(human_text))
            else:
                print("TennisRobo: ", end="")
                print(generate_response(human_text))
                article_sentences.remove(human_text)
    else:
        continue_dialogue = False
        print("TennisRobo: Good bye and take care of yourself...")

# In the script above, we first set the flag continue_dialogue to true. After that, we print a welcome message to the user asking for any input. Next, we initialize a while loop that keeps executing until the continue_dialogue flag is true. Inside the loop, the user input is received, which is then converted to lowercase. The user input is stored in the human_text variable. If the user enters the word "bye", the continue_dialogue is set to false and a goodbye message is printed to the user.

# On the other hand, if the input text is not equal to "bye", it is checked if the input contains words like "thanks", "thank you", etc. or not. If such words are found, a reply "Most welcome" is generated. Otherwise, if the user input is not equal to None, the generate_response method is called which fetches the user response based on the cosine similarity as explained in the last section.

# Once the response is generated, the user input is removed from the collection of sentences since we do not want the user input to be part of the corpus. The process continues until the user types "bye". You can see why this type of chatbot is called a rule-based chatbot. There are plenty of rules to follow and if we want to add more functionalities to the chatbot, we will have to add more rules.

# The response might not be precise, however, it still makes sense.

# It is important to mention that the idea of this article is not to develop a perfect chatbot but to explain the working principle of rule-based chatbots.


# Conclusion
# Chatbots are conversational agents that engage in different types of conversations with humans. Chatbots are finding their place in different strata of life ranging from personal assistant to ticket reservation systems and physiological therapists. Having a chatbot in place of humans can actually be very cost effective. However, developing a chatbot with the same efficiency as humans can be very complicated.

# In this article, we show how to develop a simple rule-based chatbot using cosine similarity. In the next article, we explore some other natural language processing arenas.