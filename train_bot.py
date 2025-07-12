import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# STEP 1: Load the intents JSON file
with open("e:/Python_Projects/projects/IntentChatBot/intents.json", encoding="utf-8") as file:
    data = json.load(file)

# STEP 2: Extract patterns and tags
patterns = []
tags = []
responses = {}

for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])
    
    # Map each tag to its responses
    responses[intent['tag']] = intent['responses']

# STEP 3: Convert patterns to vectors using Bag of Words
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(patterns)
y = tags

# STEP 4: Train the intent classification model
model = LogisticRegression()
model.fit(X, y)

# STEP 5: Save the trained model, vectorizer, and response map
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
pickle.dump(responses, open('responses.pkl', 'wb'))

print("✅ Training complete! Model, vectorizer, and responses are saved.")
