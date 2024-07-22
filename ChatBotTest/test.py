import bs4 as bs
import urllib.request
import re
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
#Open the cat web data page
cat_data = urllib.request.urlopen('https://simple.wikipedia.org/wiki/Cat').read()
#Find all the paragraph html from the web page
cat_data_paragraphs  = bs.BeautifulSoup(cat_data,features="html.parser").find_all('p')
#Creating the corpus of all the web page paragraphs
cat_text = ''
#Creating lower text corpus of cat paragraphs
for p in cat_data_paragraphs:
    cat_text += p.text.lower()
# With the above code, we would end up with a collection of paragraphs from the website page. Next, we need to clean up the text to remove the bracket number and empty spaces.
cat_text=re.sub(r'\s+', ' ',re.sub(r'\[[0-9]*\]', ' ', cat_text))
# print(cat_text)
# The process above would remove the bracket number from the corpus. I specifically do not remove the symbols and punctuation because it would sound natural when the conversation happens with the chatbot. Finally, I would create a list of sentences based on the corpus we have created previously.
nltk.download('punkt')
cat_sentences = nltk.sent_tokenize(cat_text)
# print(cat_sentences)
# The idea of using a list of sentences is to measure the cosine similarity between the query text we put into our chatbot to every single text in the list of sentences. Whichever result produces the closest similarity (highest cosine similarity) would become the chatbot answer. Our corpus above is still in the form of text, and cosine similarity would not accept the text data; that is why we need to develop a chatbot by transforming the corpus into a numerical vector. Common practice is transforming the text into bag-of-words (word counting) or using the TF-IDF approach (frequency probability). In our case, we would use TF-IDF. I would create a function that accepts query text and gives an output based on the cosine similarity in the following code. Let’s take a look at the code.

def chatbot_answer(user_query):
    
    #Append the query to the sentences list
    cat_sentences.append(user_query)
    #Create the sentences vector based on the list
    vectorizer = TfidfVectorizer()
    sentences_vectors = vectorizer.fit_transform(cat_sentences)
    
    #Measure the cosine similarity and take the second closest index because the first index is the user query
    vector_values = cosine_similarity(sentences_vectors[-1], sentences_vectors)
    answer = cat_sentences[vector_values.argsort()[0][-2]]
    #Final check to make sure there are result present. If all the result are 0, means the text input by us are not captured in the corpus
    input_check = vector_values.flatten()
    input_check.sort()
    
    if input_check[-2] == 0:
        return "Please Try again"
    else: 
        return answer
    
# Finally, we could create a simple chatbot using the following code.
print("Hello, I am the Cat Chatbot. What is your meow questions?:")
while(True):
    query = input().lower()
    if query not in ['bye', 'good bye', 'take care']:
        print("Cat Chatbot: ", end="")
        print(chatbot_answer(query))
        cat_sentences.remove(query)
    else:
        print("See You Again")
        break