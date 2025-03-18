import speech_recognition as sr
import pyttsx3
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="mistral")

chat_history = ChatMessageHistory()

engine = pyttsx3.init()
engine.setProperty('rate', 150)

recognizer = sr.Recognizer()

def speak(text):
    engine.say(text)
    engine.runAndWait()
    
def listen():
    with sr.Microphone() as source:
        print(" \nListening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            speak(text)
            return text.lower()
        except sr.UnknownValueError:
            print(" \nGoogle Speech Recognition could not understand audio")
            return ""
        except sr.RequestError:
            print(" \nGoogle Speech Recognition could not request results")
            return ""
prompt  = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="Previous Conversation:\n{chat_history}\n\nQuestion: {question}\nAnswer:",
)

def run_chain(question):
    chat_history_text = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in chat_history.messages])
    response = llm.invoke(prompt.format(chat_history=chat_history_text, question=question))
    chat_history.add_user_message(question)
    chat_history.add_ai_message(response)
    return response

speak("Welcome to the voice agent. I am here to help you with any questions you may have.")
while True:
    query = listen()
    if "exit" in query or 'stop' in query:
        speak("Goodbye!")
    if query:
        response = run_chain(query)
        print(f"\n Ai Response: {response}")
        speak(response)