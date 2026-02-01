from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Sample Data: Text and Labels (1 = Spam, 0 = Ham/Not Spam)
emails = [
    "Win a free iPhone now", "Hey, are we still meeting for lunch?",
    "Cash prize winner call now", "Please find the attached report",
    "Cheap loans available today", "Your Amazon order has shipped"
]
labels = [1, 0, 1, 0, 1, 0] 

# 1. Convert text into numbers (word counts)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)

# 3. Train the Naïve Bayes Model
model = MultinomialNB()
model.fit(X_train, y_train)

# 4. Predict on new data
new_email = ["Win a prize today"]
new_email_counts = vectorizer.transform(new_email)
prediction = model.predict(new_email_counts)

print(f"Prediction for '{new_email[0]}': {'Spam' if prediction[0] == 1 else 'Not Spam'}")