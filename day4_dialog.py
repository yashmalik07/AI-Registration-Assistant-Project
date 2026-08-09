import json
import random
import re
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# NLTK Setup
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

# 2. State & Context Management
class DialogManager:
    def __init__(self):
        self.state = "START"  # States: START, COLLECT_NAME, COLLECT_EMAIL, COLLECT_FIELD, COMPLETED
        self.user_data = {}

    def validate_email(self, email):
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email.strip()))

    def process_input(self, user_input):
        cleaned_input = user_input.strip()

        # State 1: Start / Greeting / Intent Recognition
        if self.state == "START":
            cleaned = preprocess_text(cleaned_input)
            input_vector = vectorizer.transform([cleaned])
            predicted_tag = model.predict(input_vector)[0]

            if predicted_tag == "register":
                self.state = "COLLECT_NAME"
                return "Sure! Let's start the registration process.\nBot: Please enter your Full Name:"
            elif predicted_tag == "greeting":
                return "Hello! Welcome to the Internship Registration Assistant. Type 'register' to start."
            elif predicted_tag == "goodbye":
                return "Goodbye! Have a nice day."
            else:
                # Fallback Response for unknown inputs
                return "I'm sorry, I didn't understand that. You can say 'Hi' or 'I want to register'."

        # State 2: Name Collection
        elif self.state == "COLLECT_NAME":
            if len(cleaned_input) >= 2 and re.match(r"^[A-Za-z\s.]+$", cleaned_input):
                self.user_data["name"] = cleaned_input
                self.state = "COLLECT_EMAIL"
                return f"Got it, {cleaned_input}!\nBot: Now, please enter your Email Address:"
            else:
                return "Invalid name! Please use letters only (e.g., Sivapriya)."

        # State 3: Email Collection & Validation
        elif self.state == "COLLECT_EMAIL":
            if self.validate_email(cleaned_input):
                self.user_data["email"] = cleaned_input
                self.state = "COLLECT_FIELD"
                return "Email verified!\nBot: Please enter your preferred field (e.g., Data Science, Web Dev, AI):"
            else:
                return "Invalid email format! Please enter a valid email (e.g., user@example.com)."

        # State 4: Field Collection & Completion
        elif self.state == "COLLECT_FIELD":
            if len(cleaned_input) >= 2:
                self.user_data["field"] = cleaned_input
                self.state = "COMPLETED"
                summary = f"\n=== Registration Completed ===\nName: {self.user_data['name']}\nEmail: {self.user_data['email']}\nField: {self.user_data['field']}\n=============================="
                return summary
            else:
                return "Please enter a valid field name."

        # State 5: Completed / Post-Registration
        elif self.state == "COMPLETED":
            return "Your registration is already complete! Type 'quit' to exit."

# 3. Conversation Loop
def start_chatbot():
    dialog = DialogManager()
    print("--- AI Registration Assistant (Type 'quit' to exit) ---")
    print("Bot: Hi there! How can I help you today?")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            print("Bot: Thank you! Goodbye.")
            break
        
        response = dialog.process_input(user_input)
        print("Bot:", response)

if __name__ == "__main__":
    start_chatbot()
