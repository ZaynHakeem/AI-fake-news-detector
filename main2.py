import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from flask import Flask, request, render_template, jsonify
import joblib

# Download the NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Clean the titles (same function)
def clean_title(title):
    title = title.lower()
    words = word_tokenize(title)
    boring_words = set(stopwords.words('english'))
    clean_words = [word for word in words if word.isalpha() and word not in boring_words]
    return ' '.join(clean_words)

# Load the saved model and TF-IDF
model = joblib.load('fake_news_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Set up Flask app
app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    article = request.json['article']
    clean_new = clean_title(article)
    new_features = tfidf.transform([clean_new])
    prediction = model.predict(new_features)[0]
    result = "That news article is Fake!" if prediction == 0 else "That news article is probably True!"
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)