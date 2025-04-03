import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Download NLTK data (run once if not already done)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Load true news dataset
true_news = pd.read_csv('True.csv')
true_news["label"] = 1

# Load fake news dataset
fake_news = pd.read_csv('Fake.csv')
fake_news["label"] = 0

# Combine the datasets
all_news = pd.concat([true_news, fake_news], ignore_index=True)

# Clean the titles
def clean_title(title):
    title = title.lower()
    words = word_tokenize(title)
    boring_words = set(stopwords.words('english'))
    clean_words = [word for word in words if word.isalpha() and word not in boring_words]
    return ' '.join(clean_words)

all_news['clean_title'] = all_news['title'].apply(clean_title)

# Turn titles into numbers
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(all_news['clean_title'])
y = all_news['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Check how good it is
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

# Save the model and TF-IDF
joblib.dump(model, 'fake_news_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
print("Model and TF-IDF saved!")