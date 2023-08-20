from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

data = {
    'reviews': [
        "I loved this product! It's amazing.",
        "This movie was terrible. I wouldn't recommend it to anyone.",
        "The service was okay, nothing special.",
        "The food at the restaurant was excellent!",_
        "I'm neutral about this book. It was neither good nor bad.",
    ],
    'labels': ['positive', 'negative', 'neutral', 'positive', 'neutral']
}

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['reviews'])

clf = MultinomialNB()
clf.fit(X, data['labels'])

user_review = input("Enter your review: ")
user_review_vec = vectorizer.transform([user_review])
predicted_label = clf.predict(user_review_vec)[0]

print("Predicted sentiment:", predicted_label)