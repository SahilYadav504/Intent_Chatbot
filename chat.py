import pickle
import random

# Load trained components
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
responses = pickle.load(open('responses.pkl', 'rb'))

print("SahilBot is ready to chat! (Type 'quit' to exit)")

while True:
    user_input = input("You: ")
    if user_input.lower().strip() == "quit":
        print("SahilBot: See you later! 👋")
        break

    # Convert user input into a vector
    input_vector = vectorizer.transform([user_input])

    # Predict the intent tag
    predicted_tag = model.predict(input_vector)[0]

    # Fetch and display a random response for that tag
    reply_list = responses.get(predicted_tag)

    if reply_list and isinstance(reply_list, list):
        bot_reply = random.choice(reply_list)
        print(f"SahilBot: {bot_reply}")
    else:
        print("SahilBot: I'm not sure I understand. Can you try rephrasing?")