import os, base64, uuid, hmac
import pandas as pd
from pathlib import Path

import streamlit as st

from langchain_community.vectorstores import AstraDB
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
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

def logout():
    for key in st.session_state.keys():
        del st.session_state[key]
    st.cache_resource.clear()
    st.cache_data.clear()
    st.rerun()

@st.cache_data()
def load_localization(locale):
    df = pd.read_csv("./customizations/localization.csv")
    df = df.query(f"locale == '{locale}'")
    return {df.key.to_list()[i]: df.value.to_list()[i] for i in range(len(df.key.to_list()))}

@st.cache_data()
def load_rails(username):
    df = pd.read_csv("./customizations/rails.csv")
    df = df.query(f"username == '{username}'")
    return {df.key.to_list()[i]: df.value.to_list()[i] for i in range(len(df.key.to_list()))}

if not check_password():
    st.stop()

username = st.session_state.user
language = st.secrets.languages[username]
lang_dict = load_localization(language)

@st.cache_resource(show_spinner=lang_dict['load_embedding'])
def load_embedding():
    return OpenAIEmbeddings()

@st.cache_resource(show_spinner=lang_dict['load_vectorstore'])
def load_vectorstore(username):
    return AstraDB(
        embedding=embedding,
        collection_name=f"vector_context_{username}",
        token=st.secrets["ASTRA_TOKEN"],
        api_endpoint=os.environ["ASTRA_ENDPOINT"],
    )

@st.cache_resource(show_spinner=lang_dict['load_message_history'])
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

def get_prompt(type):
    template = ''
    if type == 'Extended results':
        template = f"""You're a helpful AI assistant tasked to answer the user's questions.
You're friendly and you answer extensively with multiple sentences. You prefer to use bulletpoints to summarize.
If you don't know the answer, just say 'I do not know the answer'.

Use the following context to answer the question:
{{context}}

Use the following chat history to answer the question:
{{chat_history}}

Question:
{{question}}

Answer in {language}:"""
    elif type == 'Short results':
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
    elif type == 'Custom':
        template = custom_prompt
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

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Título personalizado
st.title("OPENLAB VC expert")

with st.sidebar:
    try:
        st.image(f"""./customizations/logo/{username}.svg""", use_column_width="always")
    except:
        try:
            st.image(f"""./customizations/logo/{username}.png""", use_column_width="always")
        except:
            st.image('./customizations/logo/default.svg', use_column_width="always")

    st.markdown(f"""{lang_dict['logout_caption']} :orange[{username}]""")
    if st.button(lang_dict['logout_button']):
        logout()

    st.divider()

    rails_dict = load_rails(username)
    embedding = load_embedding()
    vectorstore = load_vectorstore(username)
    chat_history = load_chat_history(username)

    disable_chat_history = st.toggle(lang_dict['disable_chat_history'])
    top_k_history = st.slider(lang_dict['k_chat_history'], 1, 50, 5, disabled=disable_chat_history)
    memory = load_memory(top_k_history if not disable_chat_history else 0)

    if st.button(lang_dict['delete_chat_history_button'], disabled=disable_chat_history):
        with st.spinner(lang_dict['deleting_chat_history']):
            memory.clear()

    disable_vector_store = st.toggle(lang_dict['disable_vector_store'])
    top_k_vectorstore = st.slider(lang_dict['top_k_vector_store'], 1, 50, 5, disabled=disable_vector_store)
    strategy = st.selectbox(lang_dict['rag_strategy'], ('Basic Retrieval', 'Maximal Marginal Relevance'), help=lang_dict['rag_strategy_help'], disabled=disable_vector_store)

    try:
        custom_prompt_text = open(f"""./customizations/prompt/{username}.txt""").read()
        custom_prompt_index = 2
    except:
        custom_prompt_text = open(f"""./customizations/prompt/default.txt""").read()
        custom_prompt_index = 0

    prompt_type = st.selectbox(lang_dict['system_prompt'], ('Short results', 'Extended results', 'Custom'), index=custom_prompt_index)
    custom_prompt = st.text_area(lang_dict['custom_prompt'], custom_prompt_text, help=lang_dict['custom_prompt_help'], disabled=(prompt_type != 'Custom'))

for message in st.session_state.messages:
    st.chat_message(message.type).markdown(message.content)

question = st.chat_input("")

if question:
    st.session_state.messages.append(HumanMessage(content=question))
    st.chat_message('human').markdown(question)

    model = load_model()
    retriever = load_retriever(top_k_vectorstore)

    relevant_documents = retriever.get_relevant_documents(query=question, k=top_k_vectorstore) if not disable_vector_store else []

    with st.chat_message('assistant'):
        content = ''
        response_placeholder = st.empty()
        history = memory.load_memory_variables({})
        chain = RunnableMap({
            'context': lambda x: x['context'],
            'chat_history': lambda x: x['chat_history'],
            'question': lambda x: x['question']
        }) | get_prompt(prompt_type) | model

        response = chain.invoke({'question': question, 'chat_history': history, 'context': relevant_documents}, config={'callbacks': [StreamHandler(response_placeholder)]})
        content += response.content
        memory.save_context({'question': question}, {'answer': content})
        st.session_state.messages.append(AIMessage(content=content))
