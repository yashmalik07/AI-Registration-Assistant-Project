import json
import random
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# NLTK தரவுகளைப் பதிவிறக்க
nltk.download('punkt')
nltk.download('wordnet')

# Lemmatizer-ஐ வரையறுக்கவும் (இங்குதான் பிழை இருந்தது)
lemmatizer = WordNetLemmatizer()

# 1. Load Intents Data
with open('intents.json', 'r') as f:
    data = json.load(f)

# 2. Text Preprocessing Function
def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(lemmatized_tokens)

patterns = []
labels = []

# Extracting Patterns & Labels
for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(preprocess_text(pattern))
        labels.append(intent['tag'])

# 3. Model Training
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)

model = LogisticRegression()
model.fit(X, labels)

# 4. Response Generator Function
def get_response(user_input):
    cleaned_input = preprocess_text(user_input)
    input_vector = vectorizer.transform([cleaned_input])
    predicted_tag = model.predict(input_vector)[0]
    
    for intent in data['intents']:
        if intent['tag'] == predicted_tag:
            return random.choice(intent['responses'])

# 5. Testing Loop
print("--- AI Assistant Active (Type 'quit' to exit) ---")
while True:
    user_query = input("You: ")
    if user_query.lower() == 'quit':
        print("Bot: Goodbye!")
        break
    response = get_response(user_query)
    print("Bot:", response)
