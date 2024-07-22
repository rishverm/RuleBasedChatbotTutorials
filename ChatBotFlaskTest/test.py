from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from flask import Flask, render_template, request

# Step 1: First, we will variables store user information.
# User_Name  = None
# User_Age  = None
# User_Job = None 

# # Step 2: Then, we will take input from the user.
# print("Hello, I'm a Chatbot \n")
# User_Name =  input("What is your name? ")
# print("How are you {0}. \n".format(User_Name))

# User_Age = input("What is your age? ")
# print("Oh, so your age is {0}. \n".format(User_Age))

# User_Job = input("What is your job profile? ")
# print("So you're a  {0}. \n".format(User_Job))

# # So now, let’s start creating a real chatbot and deploy it on Flask. We will use the ChatterBot Python library, which is mainly developed for building chatbots. What is ChatterBot, and how does it work? ChatterBot is a machine learning library that helps to generate an automatic response based on the user’s input. It uses a Natural Language Processing-based algorithm to generate repossessed based on the user’s contexts. Picture available at link: https://python.plainenglish.io/how-to-make-a-rule-based-chatbot-in-python-using-flask-d5649a6ce308

# # Then, we will name our chatbot. It can be anytime as per our need.
# chatbot=ChatBot('Verminator')
# # Step 3: We will start training our chatbot using its pre-defined dataset.
# # Create a new trainer for the chatbot
# trainer = ChatterBotCorpusTrainer(chatbot)

# # Now, let us train our bot with multiple corpus
# trainer.train("chatterbot.corpus.english.greetings",
#               "chatterbot.corpus.english.conversations" )
# # chatterbot.corpus.english.greetings and chatterbot.corpus.english.conversations are the pre-defined dataset used to train small talks and everyday conversational to our chatbot.


# # Step 4: Then, we will check how our chatbot is responding to our question using the below code.
# response = chatbot.get_response("How are you?")
# print(response)

# Open the file <Python-folder>\Lib\site-packages\sqlalchemy\util\compat.py Go to line 264 which states:
# if win32 or jython:
#     time_func = time.clock
    
# else:
#     time_func = time.time

# Change it to:


# if win32 or jython:
#     #time_func = time.clock
#     pass
# else:
#     time_func = time.time


# What is Python Flask?
# The Flask is a Python micro-framework used to create small web applications and websites using Python. Flask works on a popular templating engine called Jinja2, a web templating system combined with data sources to the dynamic web pages.

# Now start developing the Flask framework based on the above ChatterBot in the above steps.

# We have already installed the Flask in the system, so we will import the Python methods we require to run the Flask microserver.

# Then, we will initialize the Flask app by adding the below code.
#Flask initialisation
app = Flask(__name__)


# Now, we will give the name to our chatbot.
chatbot=ChatBot('Verminator')

# Add training code for a chatbot.
# Create a new trainer for the chatbot
trainer = ChatterBotCorpusTrainer(chatbot)
# Now, let us train our bot with multiple corpus
trainer.train("chatterbot.corpus.english.greetings",
              "chatterbot.corpus.english.conversations" )

# We will create the Flask decorator and a route in this step.
@app.route("/")
def index():
	return render_template("index.html")
# The route() is a function of a Flask class used to define the URL mapping associated with the function-> chooses the area the index.html file will be vreated. 
# Then we make an index function to render/make the HTML code associated with the index.html file using the render_template function-> names the file, creates code for file. 
# In the next step, we will make a response function that will take the input from the user, and also, it will return the result or response from our trained chatbot.

# Create a function to take input from the user and respond accordingly.
@app.route("/get", methods=["GET","POST"])
def chatbot_response():
	msg = request.form["msg"]
	response = chatbot.get_response(msg)
	return str(response)


# Then, we will add the final code that will start the Flask server after interpreting the whole code.
if __name__ == "__main__":
    app.run()