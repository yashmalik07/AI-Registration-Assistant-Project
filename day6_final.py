import json
import random
import re
import os
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Setup NLP
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
lemmatizer = WordNetLemmatizer()

# Load Intent Data
with open('intents.json', 'r') as f:
    data = json.load(f)

def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(lemmatized_tokens)

patterns, labels = [], []
for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(preprocess_text(pattern))
        labels.append(intent['tag'])

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)
model = LogisticRegression()
model.fit(X, labels)

# Bonus Feature: Simple Sentiment Analyzer
def analyze_sentiment(text):
    text = text.lower()
    positive_words = ['good', 'great', 'awesome', 'happy', 'thanks', 'thank']
    negative_words = ['bad', 'worst', 'angry', 'slow', 'useless', 'hate']
    
    if any(word in text for word in positive_words):
        return "POSITIVE"
    elif any(word in text for word in negative_words):
        return "NEGATIVE"
    return "NEUTRAL"

# Final Bot Engine
class FinalRegistrationBot:
    def __init__(self):
        self.state = "START"
        self.user_data = {}
        self.db_file = "registrations.json"

    def save_to_json(self):
        registrations = []
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'r') as f:
                    registrations = json.load(f)
            except json.JSONDecodeError:
                registrations = []
        registrations.append(self.user_data)
        with open(self.db_file, 'w') as f:
            json.dump(registrations, f, indent=4)

    def process_input(self, user_input):
        cleaned_input = user_input.strip()

        # Sentiment Check
        sentiment = analyze_sentiment(cleaned_input)

        if self.state == "START":
            cleaned = preprocess_text(cleaned_input)
            input_vector = vectorizer.transform([cleaned])
            predicted_tag = model.predict(input_vector)[0]

            if predicted_tag == "register":
                self.state = "COLLECT_NAME"
                return "Awesome! Let's get you registered.\nBot: What is your Full Name?"
            elif predicted_tag == "greeting":
                prefix = "Glad to see your positive energy! " if sentiment == "POSITIVE" else ""
                return f"{prefix}Hello! Welcome to the AI Internship Assistant. Type 'register' to begin."
            elif predicted_tag == "goodbye":
                return "Goodbye! Have a fantastic day ahead."
            else:
                # FAQ / Fallback response
                return "I'm still learning! You can ask to 'register' or say 'Hi'."

        elif self.state == "COLLECT_NAME":
            if len(cleaned_input) >= 2 and re.match(r"^[A-Za-z\s.]+$", cleaned_input):
                self.user_data["name"] = cleaned_input
                self.state = "COLLECT_EMAIL"
                return f"Nice to meet you, {cleaned_input}!\nBot: Please share your Email Address:"
            else:
                return "Please enter a valid name (letters only)."

        elif self.state == "COLLECT_EMAIL":
            if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', cleaned_input):
                self.user_data["email"] = cleaned_input
                self.state = "COLLECT_FIELD"
                return "Email verified successfully!\nBot: Which domain do you prefer? (e.g., Data Science, Web Dev, AI):"
            else:
                return "Invalid email format. Please try again (e.g., name@example.com)."

        elif self.state == "COLLECT_FIELD":
            if len(cleaned_input) >= 2:
                self.user_data["field"] = cleaned_input
                reg_id = f"REG{random.randint(1000, 9999)}"
                self.user_data["registration_id"] = reg_id
                self.save_to_json()
                self.state = "COMPLETED"

                return (
                    f"\n========================================\n"
                    f"🎉 REGISTRATION SUCCESSFUL!\n"
                    f"========================================\n"
                    f"Registration ID : {reg_id}\n"
                    f"Name            : {self.user_data['name']}\n"
                    f"Email           : {self.user_data['email']}\n"
                    f"Domain          : {self.user_data['field']}\n"
                    f"Sentiment       : {sentiment}\n"
                    f"========================================"
                )
            else:
                return "Please enter a valid domain name."

        elif self.state == "COMPLETED":
            return "Your registration is complete! Type 'quit' to exit."

# Main Function
def main():
    bot = FinalRegistrationBot()
    print("--- 🚀 AI Registration Assistant v2.0 (Type 'quit' to exit) ---")
    print("Bot: Hi! How can I assist you today?")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            print("Bot: Thanks for using the assistant! Goodbye.")
            break

        response = bot.process_input(user_input)
        print("Bot:", response)

if __name__ == "__main__":
    main()
