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

# 1. Load Intent Data
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

# Train Model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)
model = LogisticRegression()
model.fit(X, labels)

# 2. Registration Logic Engine
class RegistrationBot:
    def __init__(self):
        self.state = "START"
        self.user_data = {}
        self.db_file = "registrations.json"

    def validate_name(self, name):
        return len(name.strip()) >= 2 and re.match(r"^[A-Za-z\s.]+$", name.strip())

    def validate_email(self, email):
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email.strip()))

    def save_to_json(self):
        registrations = []
        # Check if file exists and load existing data
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'r') as f:
                    registrations = json.load(f)
            except json.JSONDecodeError:
                registrations = []

        registrations.append(self.user_data)

        # Save back to JSON file
        with open(self.db_file, 'w') as f:
            json.dump(registrations, f, indent=4)

    def process_input(self, user_input):
        cleaned_input = user_input.strip()

        if self.state == "START":
            cleaned = preprocess_text(cleaned_input)
            input_vector = vectorizer.transform([cleaned])
            predicted_tag = model.predict(input_vector)[0]

            if predicted_tag == "register":
                self.state = "COLLECT_NAME"
                return "Sure! Let's start the registration workflow.\nBot: Please enter your Full Name:"
            elif predicted_tag == "greeting":
                return "Hello! Welcome to the Internship Registration Assistant. Type 'register' to start."
            elif predicted_tag == "goodbye":
                return "Goodbye! Have a great day."
            else:
                return "I didn't understand that. You can say 'Hi' or 'I want to register'."

        elif self.state == "COLLECT_NAME":
            if self.validate_name(cleaned_input):
                self.user_data["name"] = cleaned_input
                self.state = "COLLECT_EMAIL"
                return f"Got it, {cleaned_input}!\nBot: Please enter your Email Address:"
            else:
                return "Invalid name format! Please use letters only."

        elif self.state == "COLLECT_EMAIL":
            if self.validate_email(cleaned_input):
                self.user_data["email"] = cleaned_input
                self.state = "COLLECT_FIELD"
                return "Email validated!\nBot: Enter your preferred field (e.g., Data Science, Web Dev, AI):"
            else:
                return "Invalid email! Example: user@example.com."

        elif self.state == "COLLECT_FIELD":
            if len(cleaned_input) >= 2:
                self.user_data["field"] = cleaned_input
                # Generate unique Registration ID
                reg_id = f"REG{random.randint(1000, 9999)}"
                self.user_data["registration_id"] = reg_id
                
                # Save Data
                self.save_to_json()
                self.state = "COMPLETED"

                confirmation = (
                    f"\n========================================\n"
                    f"   REGISTRATION CONFIRMATION SUCCESSFUL  \n"
                    f"========================================\n"
                    f"Registration ID : {reg_id}\n"
                    f"Name            : {self.user_data['name']}\n"
                    f"Email           : {self.user_data['email']}\n"
                    f"Field           : {self.user_data['field']}\n"
                    f"Status          : Saved to {self.db_file}\n"
                    f"========================================"
                )
                return confirmation
            else:
                return "Please enter a valid field name."

        elif self.state == "COMPLETED":
            return "Registration completed already! Type 'quit' to exit."

# 3. Main Loop
def main():
    bot = RegistrationBot()
    print("--- AI Registration System (Type 'quit' to exit) ---")
    print("Bot: Hello! How can I help you today?")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            print("Bot: Thank you! Goodbye.")
            break

        response = bot.process_input(user_input)
        print("Bot:", response)

if __name__ == "__main__":
    main()
