#######################
# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import pickle
#######################
# Step 1: Import Libraries and Load the Model
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
#######################

# Page configuration
st.set_page_config(
    page_title="DeepNexus Studio",
    page_icon="artificial-intelligence.png",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

#######################
# CSS styling
st.markdown("""
<style>

[data-testid="block-container"] {
    padding-left: 2rem;
    padding-right: 2rem;
    padding-top: 1rem;
    padding-bottom: 0rem;
    margin-bottom: -7rem;
}

[data-testid="stVerticalBlock"] {
    padding-left: 0rem;
    padding-right: 0rem;
}

[data-testid="stMetric"] {
    background-color: #393939;
    text-align: center;
    padding: 15px 0;
}

[data-testid="stMetricLabel"] {
  display: flex;
  justify-content: center;
  align-items: center;
}

[data-testid="stMetricDeltaIcon-Up"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

[data-testid="stMetricDeltaIcon-Down"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

</style>
""", unsafe_allow_html=True)

#######################
# Sidebar
with st.sidebar:
    st.markdown("""
        <style>
        .my-icon {
        image: url('artificial-intelligence.png');
        }
        </style>
        <div class='my-icon'></div>
        """, unsafe_allow_html=True)
    st.title('DeepNexus')

    project = ['Go to homepage','Sentiment Analysis','Next word Prediction','Text Classification']
    selected_project= st.selectbox('Select a year', project)

    color_theme_list = ['blues', 'cividis', 'greens', 'inferno', 'magma', 'plasma', 'reds', 'rainbow', 'turbo', 'viridis']
    selected_color_theme = st.selectbox('Select a color theme', color_theme_list)


#######################
if selected_project == 'Go to homepage':

    st.markdown("<h1 style='text-align: center; color: DarkSalmon; font-size:28px;'>Welcome to DeepNexus Studio!</h1>", unsafe_allow_html=True)
    #st.markdown("<h3 style='text-align: center; font-size:56px;'<p>&#129302;</p></h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: grey; font-size:20px;'>The term <b>Deep</b> signifies the use of deep learning, a branch of artificial intelligence that focuses on complex neural networks to model and understand intricate patterns in data. <b>Nexus</b> represents a connection or link, emphasizing how your application serves as a central hub that connects advanced deep learning techniques with practical natural language processing (NLP) tasks. <b>Studio</b> suggests a creative space or environment where users can interact with and explore these sophisticated technologies.Together, <b>DeepNexus Studio</b> conveys the idea of a powerful, connected platform where deep learning meets NLP, offering users an intuitive and innovative space to work with cutting-edge text analysis tools.</h3>", unsafe_allow_html=True)
    st.write("")
    st.write("")
    
    st.markdown("""
    <div style="text-align: center;">
        <a href="https://github.com/aliarmaghan">
            <img src="https://img.shields.io/github/stars/dlopezyse/Synthia.svg?logo=github&style=social" alt="Star">
        </a>
        <a href="https://x.com/armaghan78">
            <img src="https://img.shields.io/twitter/follow/armaghan78?style=social" alt="Follow">
        </a>
        <a href="https://www.buymeacoffee.com/lopezyse">
            <img src="https://img.shields.io/badge/Buy%20me%20a%20coffee--yellow.svg?logo=buy-me-a-coffee&logoColor=orange&style=social" alt="Buy me a coffee">
        </a>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('___')
    st.write(':point_left: Use the menu at left to select a task (click on > if closed).')
    st.markdown('___')
    st.markdown("<h3 style='text-align: left; color:#F63366; font-size:18px;'><b>What is this App about?<b></h3>", unsafe_allow_html=True)
    st.write("Learning happens best when content is personalized to meet our needs and strengths.")
    st.write("**DeepNexus Studio** is an innovative web application that merges the latest deep learning techniques with natural language processing (NLP) to provide powerful, real-time text analysis tools. The platform features two core functionalities:")   
    st.write("1. **Sentiment Analysis:** Utilizing a Simple Recurrent Neural Network (SimpleRNN), DeepNexus Studio analyzes and classifies the sentiment of text, particularly movie reviews, into positive or negative categories based on trained data from the IMDB dataset.") 
    st.write("2. **Next Word Prediction:** Powered by a Long Short-Term Memory (LSTM) model, the application predicts the next word in a given sequence, aiding users in completing sentences or generating text continuations that are contextually accurate and coherent.")
    st.write("**Technology Stack:** DeepNexus Studio is developed using Python, TensorFlow/Keras, Streamlit, Pandas, NumPy, and Pickle, ensuring a robust and scalable platform.") 
        
    st.markdown("<h3 style='text-align: left; color:#F63366; font-size:18px;'><b>Who is this App for?<b></h3>", unsafe_allow_html=True)
    st.write("Anyone can use this App completely for free! If you like it :heart:, show your support by sharing :+1: ")
    st.write("Are you into NLP? Our code is 100% open source and written for easy understanding. Fork it from [GitHub] (https://github.com/aliarmaghan), and pull any suggestions you may have. Become part of the community! Help yourself and help others :smiley:")

#-----------------------------------------
# ---------------------------- Sentiment Analysis --------------------------------#
elif selected_project == 'Sentiment Analysis':

    # Load the IMDB dataset word index
    word_index = imdb.get_word_index()
    reverse_word_index = {value: key for key, value in word_index.items()}

    # Load the pre-trained model with ReLU activation
    model = load_model('simple_rnn/simple_rnn_imdb.h5')

    # Step 2: Helper Functions
    # Function to decode reviews
    def decode_review(encoded_review):
        return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

    # Function to preprocess user input
    def preprocess_text(text):
        words = text.lower().split()
        encoded_review = [word_index.get(word, 2) + 3 for word in words]
        padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
        return padded_review

    # Streamlit app
    st.title('IMDB Movie Review Sentiment Analysis')
    st.write('Enter a movie review to classify it as positive or negative.')

    # User input
    user_input = st.text_area('Movie Review')

    if st.button('Classify'):

        preprocessed_input=preprocess_text(user_input)

        ## MAke prediction
        prediction=model.predict(preprocessed_input)
        sentiment='Positive' if prediction[0][0] > 0.5 else 'Negative'

        # Display the result
        st.write(f'Sentiment: {sentiment}')
        st.write(f'Prediction Score: {prediction[0][0]}')
    else:
        st.write('Please enter a movie review.')

#############################
#-----------------------------------------------------------------------------------------------#

#------------------------Next word prediction--------------------------------#
elif selected_project == 'Next word Prediction':
    
    # Load the LSTM Model
    model = load_model('lstm_rnn/next_word_lstm.h5')

    ## load the tokenizer
    with open('lstm_rnn/tokenizer.pickle','rb') as handle:
        tokenizer = pickle.load(handle)

    # Function to predict the next word
    def predict_next_word(model, tokenizer, text, max_sequence_len):
        token_list = tokenizer.texts_to_sequences([text])[0]
        if len(token_list) >= max_sequence_len:
            token_list = token_list[-(max_sequence_len-1):]  # Ensure the sequence length matches max_sequence_len-1
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted, axis=1)
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                return word
        return None
    
    # streamlit app
    st.title("Next Word Prediction With LSTM And Early Stopping")
    input_text=st.text_input("Enter the sequence of Words","To be or not to")
    if st.button("Predict Next Word"):
        max_sequence_len = model.input_shape[1] + 1  # Retrieve the max sequence length from the model input shape
        next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
        st.write(f'Next word: {next_word}')

#############################
#-----------------------------------------------------------------------------------------------#

#------------------------Text Classification--------------------------------#
elif selected_project == 'Text Classification': 
    st.write('Welcome')