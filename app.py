import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# preprocessing function
def transform_text(text):
    # word to lowercase
    text = text.lower()

    # tokenizing the text
    text = nltk.word_tokenize(text)

    # creating an empty list to store processed text
    proccesed_text = []

    # removing alphanumeric / special characters
    for word in text:
        if word.isalnum():
            proccesed_text.append(word)

    # copying processed text to text
    text = proccesed_text[:]
    proccesed_text.clear()

    # removing stopwords
    for word in text:
        if word not in stopwords.words('english') and word not in string.punctuation:
            proccesed_text.append(word)

    # stemming
    text = proccesed_text[:]
    proccesed_text.clear()
    ps = PorterStemmer()
    for word in text:
        proccesed_text.append(ps.stem(word))
    return " ".join(proccesed_text)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


st.title("Email Spam Classifier")

input_email = st.text_area("enter your mail", "")


if st.button('Predict'):
    # 1. preprocess
    st.write('Number of characters in mail :', len(input_email))
    transformed_mail = transform_text(input_email)

    # vectorize
    vector_input = tfidf.transform([transformed_mail]).toarray()
    # predict
    result = model.predict(vector_input)
    # display
    if result == 1:
        st.header('Spam')
    else:
        st.header("Not Spam ")
