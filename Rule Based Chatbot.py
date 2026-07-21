import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True) # Added to resolve LookupError

data = {
    "hello": "Hi there! How can I help you today?",
    "what is your name": "I am a simple rule-based chatbot.",
    "how does this work": "I use TF-IDF vectorization to match your input to my predefined responses.",
    "what is nlp": "NLP stands for Natural Language Processing, a field of AI focused on human-computer interaction.",
    "bye": "Goodbye! Have a great day!"
}

questions = list(data.keys())
responses = list(data.values())

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    return " ".join([stemmer.stem(w) for w in tokens if w not in stop_words and w not in string.punctuation])

processed_questions = [preprocess(q) for q in questions]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed_questions)

def get_response(user_input):
    user_vec = vectorizer.transform([preprocess(user_input)])
    similarities = cosine_similarity(user_vec, tfidf_matrix)
    best_idx = similarities.argmax()

    return responses[best_idx] if similarities[0][best_idx] > 0.1 else "I'm sorry, I don't understand that."

print("Chatbot: Hello! Type 'bye' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'bye':
        print("Chatbot: Goodbye!")
        break
    print("Chatbot:", get_response(user_input))
