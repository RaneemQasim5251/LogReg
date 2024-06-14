from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Initialize the Flask application
app = Flask(__name__)

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Function to preprocess text
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalnum() and word not in stop_words]
    return ' '.join(words)

# Load and preprocess the dataset
columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
df = pd.read_csv('C:/Users/lenovo pc/Downloads/training.1600000.processed.noemoticon.csv/training.1600000.processed.noemoticon.csv', encoding='latin-1', names=columns)
df['target'] = df['target'].map({0: 0, 4: 1})
df = df.sample(10000, random_state=42)
df['text'] = df['text'].apply(preprocess_text)

# Use TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X = vectorizer.fit_transform(df['text']).toarray()
y = df['target'].values

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression Model from Scratch
class LogisticRegressionFromScratch:
    def __init__(self, learning_rate=0.005, epochs=3000, regularization=None, lambda_=0.01):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.regularization = regularization
        self.lambda_ = lambda_
        self.weights = None

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1] + 1)
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        for epoch in range(self.epochs):
            z = np.dot(X, self.weights)
            h = sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size

            if self.regularization == 'l2':
                gradient += self.lambda_ * self.weights / y.size
            elif self.regularization == 'l1':
                gradient += self.lambda_ * np.sign(self.weights) / y.size

            self.weights -= self.learning_rate * gradient

    def predict_proba(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return sigmoid(np.dot(X, self.weights))

    def predict(self, X):
        return self.predict_proba(X) >= 0.5

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Train the model
model = LogisticRegressionFromScratch(learning_rate=0.005, epochs=3000, regularization='l2', lambda_=0.01)
model.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        tweet = request.form['tweet']
        preprocessed_tweet = preprocess_text(tweet)
        tweet_features = vectorizer.transform([preprocessed_tweet]).toarray()
        tweet_features = scaler.transform(tweet_features)
        tweet_prob = model.predict_proba(tweet_features)
        tweet_pred = model.predict(tweet_features)
        sentiment = "Positive" if tweet_pred[0] == 1 else "Negative"
        return render_template('index.html', tweet=tweet, sentiment=sentiment, probability=tweet_prob[0])

if __name__ == '__main__':
    app.run(debug=True)
