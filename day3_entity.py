import re

# User Data Structure (சேகரிக்கப்படும் தகவல்களைச் சேமிக்க)
user_profile = {
    "name": None,
    "email": None,
    "field": None
}

# Validation Functions
def validate_name(name):
    # பெயர் எழுத்துக்கள் மற்றும் இடைவெளிகளை மட்டும் கொண்டிருக்க வேண்டும்
    cleaned_name = name.strip()
    if len(cleaned_name) >= 2 and re.match(r"^[A-Za-z\s.]+$", cleaned_name):
        return True
    return False

def validate_email(email):
    # Regex Email Format Verification
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if re.match(pattern, email.strip()):
        return True
    return False

# Entity Extraction & Collection Engine
def collect_user_details():
    print("--- Day 3: Entity Extraction & Validation ---")
    
    # 1. Name Extraction & Validation
    while True:
        name_input = input("Bot: What is your full name?\nYou: ")
        if validate_name(name_input):
            user_profile["name"] = name_input.strip()
            print("Bot: Name recorded successfully!\n")
            break
        else:
            print("Bot: Invalid name! Please enter letters only (at least 2 characters).\n")

    # 2. Email Extraction & Validation
    while True:
        email_input = input("Bot: Please enter your email address:\nYou: ")
        if validate_email(email_input):
            user_profile["email"] = email_input.strip()
            print("Bot: Email verified successfully!\n")
            break
        else:
            print("Bot: Invalid email format! (Example: user@example.com)\n")

    # 3. Field Selection
    while True:
        field_input = input("Bot: Enter your preferred field (e.g., Data Science, Web Dev, AI):\nYou: ")
        if len(field_input.strip()) >= 2:
            user_profile["field"] = field_input.strip()
            print("Bot: Field saved successfully!\n")
            break
        else:
            print("Bot: Please enter a valid field name.\n")

    # Display Extracted Data Structure
    print("=" * 40)
    print("Extracted Entity Data (Stored Dictionary):")
    print("=" * 40)
    for key, value in user_profile.items():
        print(f"• {key.capitalize()}: {value}")
    print("=" * 40)

# Run Task
if __name__ == "__main__":
    collect_user_details()
