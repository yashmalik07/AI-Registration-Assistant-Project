# AI Registration Assistant Chatbot

An AI-powered conversational chatbot built using Python, NLTK, and Scikit-Learn that collects, validates, and stores user registration data for internship programs.

## Features
- **Intent Recognition:** Identifies greetings, registration requests, and exit commands using Logistic Regression.
- **Entity Extraction & Validation:** Validates full name and email formats using Regex.
- **Dialog Management:** Manages state transitions smoothly through the entire conversation flow.
- **JSON Storage:** Automatically generates a unique Registration ID and saves details to `registrations.json`.
- **Sentiment Analysis:** Detects user tone during the chat.

## Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone <YOUR_GITHUB_REPOSITORY_URL>
   cd "project no.1"
   ```

2. **Install Required Libraries:**
   ```bash
   pip install nltk scikit-learn
   ```

3. **Run the Application:**
   ```bash
   python day6_final.py
   ```

## File Structure
- `intents.json`: Training dataset for chatbot intents.
- `day2_intent.py`: NLP tokenization, lemmatization, and intent classification.
- `day3_entity.py`: Data extraction and regex validation rules.
- `day4_dialog.py`: Dialog state engine and conversation loop.
- `day5_registration.py`: Data persistence and unique ID generation.
- `day6_final.py`: Combined main application with sentiment analysis.
- `registrations.json`: Generated JSON database storing output details.
-
