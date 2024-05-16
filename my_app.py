import streamlit as st
import cohere
from dotenv import load_dotenv
import os

load_dotenv()
co = cohere.Client(os.getenv('COHERE_API_KEY')) 

message = []

def generate_answer(input):
    if len(input) == 0:
        return None
    response = co.generate(
    model = 'command',
    prompt= 'Show only the translation of the giving french phrase into english \n--\nPost: {}'.format(input),
    max_tokens=20,
    temperature=0.5,
    k=0,
    p=1,
    frequency_penalty=0, 
    presence_penalty=0, 
    stop_sequences=["--"], 
    return_likelihoods='NONE'
    )
    st.write(response.generations[0].text)
    
input = "Bienvenus dans le monde"

st.title('Traduction AI en Anglais')

user_question = st.text_area('Entrer la phrase: ', height=100)
st.button('Générer', on_click = generate_answer(user_question))
