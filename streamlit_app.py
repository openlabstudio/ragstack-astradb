import os, base64
from pathlib import Path
import hmac
import tempfile
import pandas as pd
import uuid

import streamlit as st

from langchain_community.vectorstores import AstraDB
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import AstraDBChatMessageHistory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, WebBaseLoader
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap
from langchain.schema import StrOutputParser

from langchain.callbacks.base import BaseCallbackHandler

import openai # openai.chat.completions.create

# --- Global Initializations (Minimal) ---
lang_dict = {"assistant_welcome": "Welcome! Please log in.", "logout_caption": "User", "logout_button": "Logout", "assistant_question":"Your question..."}
language = "es_ES"
rails_dict = {}
embedding = None
vectorstore = None
chat_history = None
memory = None
disable_vector_store = False
strategy = 'Basic Retrieval'
prompt_type = 'Short results'
custom_prompt = ''

print("Streamlit App Top Level Execution")

st.set_page_config(page_title='Your Enterprise Sidekick', page_icon='🚀')

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "debug_messages" not in st.session_state:
    st.session_state.debug_messages = []

# --- Streaming Callback Handler ---
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▌")

# --- Helper Functions ---
def check_password():
    def login_form():
        with st.form("credentials"):
            st.text_input('Username', key='username_input')
            st.text_input('Password', type='password', key='password_input')
            st.form_submit_button('Login', on_click=password_entered)

    def password_entered():
        username_val = st.session_state.get('username_input', '')
        password_val = st.session_state.get('password_input', '')
        
        if 'passwords' in st.secrets and \
           username_val in st.secrets['passwords'] and \
           hmac.compare_digest(password_val, st.secrets.passwords[username_val]):
            st.session_state['password_correct'] = True
            st.session_state.user = username_val
            if 'password_input' in st.session_state: del st.session_state['password_input']
            if 'username_input' in st.session_state: del st.session_state['username_input']
        else:
            st.session_state['password_correct'] = False

    if st.session_state.get('password_correct', False):
        return True
    login_form()
    if "password_correct" in st.session_state and not st.session_state['password_correct']:
        st.error(lang_dict.get('login_error', '😕 User not known or password incorrect'))
    return False

def logout():
    keys_to_delete = list(st.session_state.keys())
    for key in keys_to_delete:
        del st.session_state[key]
    st.cache_resource.clear()
    st.cache_data.clear()
    st.rerun()

def vectorize_text(uploaded_files_param, current_vectorstore, current_lang_dict):
    if not current_vectorstore:
        st.error(current_lang_dict.get('vectorstore_not_ready_admin', "Vectorstore not ready for admin upload."))
        return
    for uploaded_file in uploaded_files_param:
        if uploaded_file is not None:
            with tempfile.TemporaryDirectory() as temp_dir_path:
                temp_dir = Path(temp_dir_path)
                file_name = uploaded_file.name
                temp_filepath = temp_dir / file_name
                with open(temp_filepath, 'wb') as f:
                    f.write(uploaded_file.getvalue())
                
                st.session_state.debug_messages.append(f"Admin: Processing file {file_name}")
                try:
                    if file_name.endswith('.txt'):
                        with open(temp_filepath, 'r', encoding='utf-8') as f_txt:
                            file_content_list = [f_txt.read()]
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
                        texts = text_splitter.create_documents(file_content_list, [{'source': file_name}] * len(file_content_list))
                        current_vectorstore.add_documents(texts)
                        st.info(f"{len(texts)} {current_lang_dict.get('load_text', 'text segments loaded')}")
                    elif file_name.endswith('.pdf'):
                        loader = PyPDFLoader(str(temp_filepath))
                        docs = loader.load()
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
                        pages = text_splitter.split_documents(docs)
                        current_vectorstore.add_documents(pages)  
                        st.info(f"{len(pages)} {current_lang_dict.get('load_pdf', 'PDF pages/segments loaded')}")
                    elif file_name.endswith('.csv'):
                        loader = CSVLoader(str(temp_filepath), encoding='utf-8')
                        docs = loader.load()
                        current_vectorstore.add_documents(docs)
                        st.info(f"{len(docs)} {current_lang_dict.get('load_csv', 'CSV documents/rows loaded')}")
                except Exception as e:
                    st.error(f"Error processing file {file_name}: {e}")
                    st.session_state.debug_messages.append(f"Admin: Error processing {file_name}: {e}")

def vectorize_url(urls_list_param, current_vectorstore, current_lang_dict):
    if not current_vectorstore:
        st.error(current_lang_dict.get('vectorstore_not_ready_admin', "Vectorstore not ready for admin URL load."))
        return
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    for url_item in urls_list_param:
        url = url_item.strip()
        if not url: continue
        try:
            st.session_state.debug_messages.append(f"Admin: Loading from URL: {url}")
            loader = WebBaseLoader(url)
            docs = loader.load()    
            pages = text_splitter.split_documents(docs)
            current_vectorstore.add_documents(pages)  
            st.info(f"{len(pages)} {current_lang_dict.get('url_pages_loaded', 'pages loaded from URL')}")
        except Exception as e:
            st.error(f"{current_lang_dict.get('url_error', 'Error loading from URL')} {url}: {e}")
            st.session_state.debug_messages.append(f"Admin: Error loading URL {url}: {e}")

def get_prompt(prompt_type_param, current_custom_prompt_text, current_language_code):
    template = ''
    lang_code_for_prompt = current_language_code if isinstance(current_language_code, str) else "the user's language"
    base_instructions = f"""You're a helpful AI assistant tasked to answer the user's questions.
If the question states the name of the user, just say 'Thanks, I'll use this information going forward'.
If you don't know the answer, just say 'I do not know the answer'.

Use the following context to answer the question:
{{context}}

Use the following chat history to answer the question:
{{chat_history}}

Question:
{{question}}

Answer in {lang_code_for_prompt}:"""

    if prompt_type_param == 'Extended results':
        print ("Prompt type: Extended results")
        template = f"""You're friendly and you answer extensively with multiple sentences. You prefer to use bulletpoints to summarize.
{base_instructions}"""
    elif prompt_type_param == 'Short results':
        print ("Prompt type: Short results")
        template = f"""You answer in an exceptionally brief way.
{base_instructions}"""
    elif prompt_type_param == 'Custom':
        print ("Prompt type: Custom")
        template = str(current_custom_prompt_text) if current_custom_prompt_text else base_instructions 
    else: 
        template = base_instructions
    return ChatPromptTemplate.from_messages([("system", template)])

def describeImage(image_bin_param, lang_param): # lang_param is the language code
    print ("describeImage")
    if not (st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")): # Check if key is actually available
        st.error("OpenAI API Key not configured. Cannot describe image.")
        return None
    image_base64 = base64.b64encode(image_bin_param).decode()
    try:
        # Ensure openai client is initialized if not already global, or pass key
        client = openai.OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY")))
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": f"Provide a search text for the main topic of the image writen in {lang_param}"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ]}], max_tokens=100)
        print (f"describeImage result: {response}")
        return response
    except Exception as e:
        st.error(f"Error describing image with OpenAI Vision: {e}")
        return None

# --- Data Loading Functions (cached) ---
@st.cache_data()
def load_localization(locale_param):
    print(f"load_localization for: {locale_param}")
    try:
        df = pd.read_csv("./customizations/localization.csv", encoding='utf-8')
        df_lang = df.query(f"locale == '{locale_param}'")
        if df_lang.empty and locale_param != "en_US":
            print(f"Locale {locale_param} not found in localization.csv, falling back to en_US")
            df_lang = df.query("locale == 'en_US'")
        if df_lang.empty:
             print("Critical: en_US locale not found. Using hardcoded minimal defaults.")
             return {"assistant_welcome": "Welcome!", "logout_caption": "User", "logout_button": "Logout", "assistant_question":"Your question...", "login_error": "User not known or password incorrect."}
        return pd.Series(df_lang.value.values, index=df_lang.key).to_dict()
    except FileNotFoundError:
        print("CRITICAL ERROR: localization.csv not found. App will likely fail or have missing text.")
        return {"assistant_welcome": "Welcome! (Localization file missing)", "logout_caption": "User", "logout_button": "Logout", "assistant_question":"Your question...", "login_error": "User not known or password incorrect."}
    except Exception as e:
        print(f"Error loading localization.csv: {e}. Using hardcoded minimal defaults.")
        return {"assistant_welcome": "Welcome! (Error in localization)", "logout_caption": "User", "logout_button": "Logout", "assistant_question":"Your question...", "login_error": "User not known or password incorrect."}

@st.cache_data()
def load_rails(username_param):
    print(f"load_rails for: {username_param}")
    try:
        df = pd.read_csv("./customizations/rails.csv", encoding='utf-8')
        df_user = df.query(f"username == '{username_param}'")
        if df_user.empty: return {}
        return pd.Series(df_user.value.values, index=df_user.key).to_dict()
    except FileNotFoundError:
        print("Warning: rails.csv not found. No rails will be loaded.")
        return {}
    except Exception as e:
        print(f"Error loading rails for {username_param} from rails.csv: {e}")
        return {}

# --- Resource Loading Functions (cached) ---
@st.cache_resource(show_spinner=lambda: lang_dict.get('load_embedding', 'Loading Embeddings...'))
def load_embedding_cached():
    print("Attempting to load OpenAIEmbeddings")
    openai_api_key = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY"))
    if not openai_api_key:
        st.error("OpenAI API Key not found in secrets or environment. Embeddings will not load.")
        return None
    try:
        return OpenAIEmbeddings(openai_api_key=openai_api_key)
    except Exception as e:
        st.error(f"Failed to load OpenAI Embeddings: {e}")
        return None

@st.cache_resource(show_spinner=lambda: lang_dict.get('load_vectorstore', 'Loading Vector Store...'))
def load_vectorstore_cached(_embedding_instance, username_key):
    print(f"Attempting to load Vector Store for {username_key}")
    if not _embedding_instance:
        st.error("Embeddings not available, cannot initialize Vector Store.")
        return None
    try:
        astra_ep = st.secrets.get("ASTRA_ENDPOINT", os.environ.get("ASTRA_ENDPOINT"))
        astra_token_val = st.secrets.get("ASTRA_TOKEN", os.environ.get("ASTRA_TOKEN"))
        astra_keyspace = st.secrets.get("ASTRA_KEYSPACE") # Allow keyspace from secrets

        if not all([astra_ep, astra_token_val]):
            st.error("Astra DB Endpoint or Token not found in secrets or environment variables.")
            return None

        return AstraDB(
            embedding=_embedding_instance,
            collection_name=f"vector_context_{username_key}",
            api_endpoint=astra_ep,
            token=astra_token_val,
            namespace=astra_keyspace, # Use namespace for keyspace
        )
    except Exception as e:
        st.error(f"Failed to initialize AstraDB Vector Store for collection 'vector_context_{username_key}': {e}")
        return None

@st.cache_resource(show_spinner=lambda: lang_dict.get('load_message_history', 'Loading Chat History...'))
def load_chat_history_cached(username_key, session_id_val):
    print(f"Attempting to load Chat History for {username_key}_{session_id_val}")
    try:
        astra_ep = st.secrets.get("ASTRA_ENDPOINT", os.environ.get("ASTRA_ENDPOINT"))
        astra_token_val = st.secrets.get("ASTRA_TOKEN", os.environ.get("ASTRA_TOKEN"))
        astra_keyspace = st.secrets.get("ASTRA_KEYSPACE")

        if not all([astra_ep, astra_token_val]):
            st.error("Astra DB Endpoint or Token not found for Chat History.")
            return None

        return AstraDBChatMessageHistory(
            session_id=f"{username_key}_{session_id_val}",
            api_endpoint=astra_ep,
            token=astra_token_val,
            keyspace_name=astra_keyspace, # Use keyspace_name for history
        )
    except Exception as e:
        st.error(f"Failed to initialize AstraDB Chat Message History: {e}")
        return None

@st.cache_resource()
def load_model_llm_cached():
    print("Attempting to load ChatOpenAI LLM")
    openai_api_key = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY"))
    if not openai_api_key:
        st.error("OpenAI API Key not found in secrets or environment. LLM will not load.")
        return None
    try:
        return ChatOpenAI(
            temperature=0.3, model='gpt-4o', streaming=True, verbose=False, openai_api_key=openai_api_key
        )
    except Exception as e:
        st.error(f"Failed to load ChatOpenAI model: {e}")
        return None

@st.cache_resource()
def load_memory_for_chat(_chat_history_instance, top_k_hist_val):
    print(f"Attempting to load Memory with top_k={top_k_hist_val}")
    if not _chat_history_instance:
        st.warning("Chat history object not available, cannot initialize full memory.")
        return None 
    return ConversationBufferWindowMemory(
        chat_memory=_chat_history_instance, return_messages=True, k=top_k_hist_val,
        memory_key="chat_history", input_key="question", output_key='answer'
    )

# --- Main Script Execution Starts After Login Check ---
if not check_password():
    st.stop()

# --- Initialize Globals (Post-Login) ---
username = st.session_state.user # This is set in check_password()
language = st.secrets.get("languages", {}).get(username, "es_ES") # Default to es_ES if not found
st.session_state.language = language # Ensure language is in session state for other functions
lang_dict = load_localization(language)

embedding = load_embedding_cached()
if embedding:
    vectorstore = load_vectorstore_cached(embedding, username)
if vectorstore:
    chat_history = load_chat_history_cached(username, st.session_state.session_id)
model = load_model_llm_cached()

# --- Session State for Messages ---
if 'messages' not in st.session_state:
    st.session_state.messages = [AIMessage(content=lang_dict.get('assistant_welcome', "Welcome!"))]

# --- Main Page Content ---
try:
    welcome_file_path = Path(f"./customizations/welcome/{username}.md")
    default_welcome_file_path = Path('./customizations/welcome/default.md')
    if welcome_file_path.is_file():
        st.markdown(welcome_file_path.read_text(encoding='utf-8'))
    elif default_welcome_file_path.is_file():
        st.markdown(default_welcome_file_path.read_text(encoding='utf-8'))
    else:
        st.markdown(lang_dict.get('assistant_welcome', "Welcome!"))
except Exception as e:
    st.warning(f"Could not load welcome message: {e}")
    st.markdown(lang_dict.get('assistant_welcome', "Welcome!"))


# --- Sidebar ---
with st.sidebar:
    # Logo
    user_logo_svg = Path(f"./customizations/logo/{username}.svg")
    user_logo_png = Path(f"./customizations/logo/{username}.png")
    default_logo_svg = Path('./customizations/logo/default.svg')
    logo_to_display = None
    if user_logo_svg.is_file(): logo_to_display = str(user_logo_svg)
    elif user_logo_png.is_file(): logo_to_display = str(user_logo_png)
    elif default_logo_svg.is_file(): logo_to_display = str(default_logo_svg)
    if logo_to_display: st.image(logo_to_display, use_column_width="always")
    else: st.text("Logo")
    st.text('')

    # Logout
    st.markdown(f"""{lang_dict.get('logout_caption', "Logged in as")} :orange[{username}]""")
    if st.button(lang_dict
                
