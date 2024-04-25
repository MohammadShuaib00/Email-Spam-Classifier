import streamlit as st 
import pickle
import string
import nltk
import sklearn 

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    # Lowercase the text
    text = text.lower()
    
    # Tokenize the text
    text = nltk.word_tokenize(text)
    
    # Remove non-alphanumeric tokens
    text = [word for word in text if word.isalnum()]
    
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word not in stop_words and word not in string.punctuation]
    
    # Stem the remaining tokens
    text = [ps.stem(word) for word in text]
    
    # Join the stemmed tokens into a single string
    return " ".join(text)


tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("SMS/Email Spam Classifier")
input_message = st.text_input("Enter the message...")

if st.button("Predict"):
    # Preprocess the input message
    transformed_message = transform_text(input_message)

    # Vectorize the preprocessed message
    vector_input = tfidf.transform([transformed_message])

    # Predict
    response = model.predict(vector_input)[0]

    # Display the result
    if response == 1:
        st.write("Spam")
    else:
        st.write("Not Spam")
    
