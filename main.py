import streamlit as st
import speech_recognition as sr
import pyttsx3
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="mistral")

if ChatMessageHistory  not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()
    
engine = pyttsx3.init()
engine.setProperty('rate', 150)

recognizer = sr.Recognizer()

def speak(text):
    engine.say(text)
    engine.runAndWait()

def listen():
    with sr.Microphone() as source:
        st.write(" \nListening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            st.write(f"speak(text):{text}")
            return text.lower()
        except sr.UnknownValueError:
            st.write(" \nGoogle Speech Recognition could not understand audio")
            return ""
        except sr.RequestError:
            st.write(" \nGoogle Speech Recognition could not request results")
            return ""


prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="Previous Conversation:\n{chat_history}\n\nQuestion: {question}\nAnswer:",
)

def run_chain(question):
    chat_history_text = "\n".join(f"{msg.type.capitalize()}: {msg.content} " for msg in st.session_state.chat_history.messages)
    response = llm.invoke(prompt.format(chat_history=chat_history_text, question=question))
    
    st.session_state.chat_history.add_user_message(question)
    st.session_state.chat_history.add_ai_message(response)
    return response

st.title(" AI Voice Assistant")
st.write(" Click the button below to speak to your Ai Assistant")

if st.button(" Start Listening"):
    user_query = listen()
    if user_query:
        ai_response = run_chain(user_query)
        st.write(f"\n Ai Response: {ai_response}")
        st.write(f"speak(ai_response):{ai_response}")
        speak(ai_response)

st.header("Chat History")
for msg in st.session_state.chat_history.messages:
    st.write(f"{msg.type.capitalize()}: {msg.content}")