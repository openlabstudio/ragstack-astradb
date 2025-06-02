import streamlit as st
import uuid
import hmac
import os
import pandas as pd
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import AstraDB
from langchain.memory import ConversationBufferWindowMemory, AstraDBChatMessageHistory
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap
from langchain.callbacks.base import BaseCallbackHandler

# Configuración inicial
st.set_page_config(page_title='OPENLAB VC expert', page_icon='🚀')

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4()

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▌")

def check_password():
    def login_form():
        with st.form("credentials"):
            st.text_input('Username', key='username')
            st.text_input('Password', type='password', key='password')
            st.form_submit_button('Login', on_click=password_entered)

    def password_entered():
        if st.session_state['username'] in st.secrets['passwords'] and hmac.compare_digest(
            st.session_state['password'], st.secrets.passwords[st.session_state['username']]
        ):
            st.session_state['password_correct'] = True
            st.session_state.user = st.session_state['username']
            del st.session_state['password']
        else:
            st.session_state['password_correct'] = False

    if st.session_state.get('password_correct', False):
        return True

    login_form()
    if "password_correct" in st.session_state:
        st.error('😕 User not known or password incorrect')
    return False

if not check_password():
    st.stop()

username = st.session_state.user
language = "es"

@st.cache_resource()
def load_embedding():
    return OpenAIEmbeddings()

@st.cache_resource()
def load_vectorstore(username):
    return AstraDB(
        embedding=embedding,
        collection_name=f"vector_context_{username}",
        token=st.secrets["ASTRA_TOKEN"],
        api_endpoint=os.environ["ASTRA_ENDPOINT"],
    )

@st.cache_resource()
def load_chat_history(username):
    return AstraDBChatMessageHistory(
        session_id=f"{username}_{st.session_state.session_id}",
        api_endpoint=os.environ["ASTRA_ENDPOINT"],
        token=st.secrets["ASTRA_TOKEN"],
    )

@st.cache_resource()
def load_memory(top_k_history):
    return ConversationBufferWindowMemory(
        chat_memory=chat_history,
        return_messages=True,
        k=top_k_history,
        memory_key="chat_history",
        input_key="question",
        output_key='answer',
    )

def get_prompt():
    template = f"""You're a helpful AI assistant tasked to answer the user's questions.
You answer in an exceptionally brief way.
If you don't know the answer, just say 'I do not know the answer'.

Use the following context to answer the question:
{{context}}

Use the following chat history to answer the question:
{{chat_history}}

Question:
{{question}}

Answer in {language}:"""
    return ChatPromptTemplate.from_messages([("system", template)])

def load_model():
    return ChatOpenAI(
        temperature=0.3,
        model='gpt-4-1106-preview',
        streaming=True,
        verbose=True
    )

def load_retriever(top_k_vectorstore):
    return vectorstore.as_retriever(search_kwargs={"k": top_k_vectorstore})

# Inicialización
embedding = load_embedding()
vectorstore = load_vectorstore(username)
chat_history = load_chat_history(username)
memory = load_memory(5)  # Puedes ajustar el valor de k según tus necesidades

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Título
st.title("OPENLAB VC expert")

# Mostrar mensajes anteriores
for message in st.session_state.messages:
    st.chat_message(message.type).markdown(message.content)

# Entrada de usuario
question = st.chat_input("")

if question:
    st.session_state.messages.append(HumanMessage(content=question))
    st.chat_message('human').markdown(question)

    model = load_model()
    retriever = load_retriever(5)  # Puedes ajustar el valor de k según tus necesidades

    relevant_documents = retriever.get_relevant_documents(query=question, k=5)

    with st.chat_message('assistant'):
        content = ''
        response_placeholder = st.empty()
        history = memory.load_memory_variables({})
        chain = RunnableMap({
            'context': lambda x: x['context'],
            'chat_history': lambda x: x['chat_history'],
            'question': lambda x: x['question']
        }) | get_prompt() | model

        response = chain.invoke({'question': question, 'chat_history': history, 'context': relevant_documents}, config={'callbacks': [StreamHandler(response_placeholder)]})
        content += response.content
        memory.save_context({'question': question}, {'answer': content})
        st.session_state.messages.append(AIMessage(content=content))
