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
# These will be properly initialized after login and lang_dict is available
lang_dict = {"assistant_welcome": "Welcome! Please log in.", "logout_caption": "User", "logout_button": "Logout", "assistant_question":"Your question..."} # Minimal defaults
language = "es_ES" # Default language
rails_dict = {}
embedding = None
vectorstore = None
chat_history = None
memory = None
disable_vector_store = False # Will be set from sidebar
strategy = 'Basic Retrieval' # Will be set from sidebar
prompt_type = 'Short results' # Will be set from sidebar
custom_prompt = '' # Will be set from sidebar

print("Streamlit App Top Level Execution") # For Replit console debugging

st.set_page_config(page_title='Your Enterprise Sidekick', page_icon='🚀')

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4()) # Ensure it's a string

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
            st.text_input('Username', key='username_input') # Use different keys for form inputs
            st.text_input('Password', type='password', key='password_input')
            st.form_submit_button('Login', on_click=password_entered)

    def password_entered():
        username_val = st.session_state.get('username_input', '')
        password_val = st.session_state.get('password_input', '')
        
        # Ensure 'passwords' secret exists and username_val is a key in it
        if 'passwords' in st.secrets and \
           username_val in st.secrets['passwords'] and \
           hmac.compare_digest(password_val, st.secrets.passwords[username_val]):
            st.session_state['password_correct'] = True
            st.session_state.user = username_val # Store the successfully logged-in user
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
            with tempfile.TemporaryDirectory() as temp_dir_path: # Use path object correctly
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
                        loader = PyPDFLoader(str(temp_filepath)) # PyPDFLoader expects string path
                        docs = loader.load()
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
                        pages = text_splitter.split_documents(docs)
                        current_vectorstore.add_documents(pages)  
                        st.info(f"{len(pages)} {current_lang_dict.get('load_pdf', 'PDF pages/segments loaded')}")
                    elif file_name.endswith('.csv'):
                        loader = CSVLoader(str(temp_filepath), encoding='utf-8') # CSVLoader expects string path
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
    template = '' # Default empty template
    # Ensure current_language_code is a string, fallback if None
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
        # Ensure custom_prompt_text is a string
        template = str(current_custom_prompt_text) if current_custom_prompt_text else base_instructions 
    else: # Fallback to a default if prompt_type_param is unexpected
        template = base_instructions
        
    return ChatPromptTemplate.from_messages([("system", template)])

def describeImage(image_bin_param, lang_param):
    print ("describeImage")
    if not openai.api_key:
        st.error("OpenAI API Key not configured for image description.")
        return None
    image_base64 = base64.b64encode(image_bin_param).decode()
    try:
        response = openai.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": f"Provide a search text for the main topic of the image writen in {lang_param}"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ]}], max_tokens=100) # Reduced for search text
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
        if df_lang.empty and locale_param != "en_US": # Fallback
            print(f"Locale {locale_param} not found in localization.csv, falling back to en_US")
            df_lang = df.query("locale == 'en_US'")
        if df_lang.empty: # Critical fallback
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
        if df_user.empty:
            return {} # No rails for this user
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
    try:
        # Ensure OPENAI_API_KEY is available from secrets
        if "OPENAI_API_KEY" not in st.secrets or not st.secrets["OPENAI_API_KEY"]:
            st.error("OpenAI API Key not found in secrets.toml. Embeddings will not load.")
            return None
        return OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
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
        # astra_keyspace = st.secrets.get("ASTRA_KEYSPACE") # Optional

        if not all([astra_ep, astra_token_val]):
            st.error("Astra DB Endpoint or Token not found in secrets.toml or environment variables.")
            return None

        return AstraDB(
            embedding=_embedding_instance,
            collection_name=f"vector_context_{username_key}",
            api_endpoint=astra_ep,
            token=astra_token_val,
            # namespace=astra_keyspace, # Add if you use a specific keyspace
        )
    except Exception as e:
        st.error(f"Failed to initialize AstraDB Vector Store for collection 'vector_context_{username_key}': {e}")
        return None

@st.cache_resource(show_spinner=lambda: lang_dict.get('load_message_history', 'Loading Chat History...'))
def load_chat_history_cached(username_key, session_id_val): # Pass session_id
    print(f"Attempting to load Chat History for {username_key}_{session_id_val}")
    try:
        astra_ep = st.secrets.get("ASTRA_ENDPOINT", os.environ.get("ASTRA_ENDPOINT"))
        astra_token_val = st.secrets.get("ASTRA_TOKEN", os.environ.get("ASTRA_TOKEN"))
        # astra_keyspace = st.secrets.get("ASTRA_KEYSPACE")

        if not all([astra_ep, astra_token_val]):
            st.error("Astra DB Endpoint or Token not found for Chat History.")
            return None

        return AstraDBChatMessageHistory(
            session_id=f"{username_key}_{session_id_val}",
            api_endpoint=astra_ep,
            token=astra_token_val,
            # keyspace_name=astra_keyspace, # Add if you use a specific keyspace
        )
    except Exception as e:
        st.error(f"Failed to initialize AstraDB Chat Message History: {e}")
        return None

@st.cache_resource()
def load_model_llm_cached(): # Renamed from load_model to avoid conflict
    print("Attempting to load ChatOpenAI LLM")
    try:
        if "OPENAI_API_KEY" not in st.secrets or not st.secrets["OPENAI_API_KEY"]:
            st.error("OpenAI API Key not found in secrets.toml. LLM will not load.")
            return None
        return ChatOpenAI(
            temperature=0.3, 
            model='gpt-4o', # Using gpt-4o
            streaming=True, 
            verbose=False, 
            openai_api_key=st.secrets["OPENAI_API_KEY"]
        )
    except Exception as e:
        st.error(f"Failed to load ChatOpenAI model: {e}")
        return None

@st.cache_resource()
def load_memory_for_chat(_chat_history_instance, top_k_hist_val): # Renamed from load_memory_cached
    print(f"Attempting to load Memory with top_k={top_k_hist_val}")
    if not _chat_history_instance:
        st.warning("Chat history object not available, cannot initialize full memory.")
        # Return a dummy or base memory if needed, or handle None upstream
        return None 
    return ConversationBufferWindowMemory(
        chat_memory=_chat_history_instance, return_messages=True, k=top_k_hist_val,
        memory_key="chat_history", input_key="question", output_key='answer'
    )

# --- Main Script Execution Starts After Login Check ---
if not check_password():
    st.stop()

# --- Initialize Globals (Post-Login) ---
username = st.session_state.user
language = st.secrets.get("languages", {}).get(username, "es_ES")
st.session_state.language = language # Make it available for other functions
lang_dict = load_localization(language) # Load language dict for the logged-in user

# Initialize resources that depend on secrets and username
embedding = load_embedding_cached()
if embedding:
    vectorstore = load_vectorstore_cached(embedding, username)
if vectorstore: # Only load chat history if vectorstore (and thus Astra creds) are OK
    chat_history = load_chat_history_cached(username, st.session_state.session_id)
model = load_model_llm_cached() # Load the main LLM

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
    if st.button(lang_dict.get('logout_button', "Logout")):
        logout()
    st.divider()

    # Rails (Suggestions)
    rails_dict = load_rails(username)
    st.subheader(lang_dict.get('rails_1', "Suggestions"))
    st.caption(lang_dict.get('rails_2', "Try asking:"))
    if rails_dict:
        for i in sorted(rails_dict.keys()): st.markdown(f"{i}. {rails_dict[i]}")
    else:
        st.markdown(lang_dict.get('no_rails_available', "No suggestions available."))
    st.divider()

    # Options Panel
    st.subheader(lang_dict.get('options_header', "Chat Options"))
    disable_chat_history = st.toggle(lang_dict.get('disable_chat_history', "Disable Chat History"), key="ch_toggle_key")
    
    default_k_history = 5 # Default value
    if 'DEFAULT_TOP_K_HISTORY' in st.secrets and isinstance(st.secrets['DEFAULT_TOP_K_HISTORY'], dict):
        default_k_history = st.secrets['DEFAULT_TOP_K_HISTORY'].get(username, 5)
    top_k_history = st.slider(lang_dict.get('k_chat_history', "K for Chat History"), 1, 10, default_k_history, disabled=disable_chat_history, key="ch_slider_key")

    if chat_history: # Initialize memory only if chat_history is valid
        memory = load_memory_for_chat(chat_history, top_k_history if not disable_chat_history else 0)
    else:
        memory = None # Ensure memory is None if chat_history failed

    if memory: # Only show delete button if memory (and thus chat_history) is available
        if st.button(lang_dict.get('delete_chat_history_button', "Delete Chat History"), disabled=disable_chat_history, key="del_ch_button_key"):
            with st.spinner(lang_dict.get('deleting_chat_history', "Deleting...")):
                memory.clear()
            st.session_state.messages = [AIMessage(content=lang_dict.get('assistant_welcome', "Welcome!"))] # Reset messages
            st.rerun()

    disable_vector_store = st.toggle(lang_dict.get('disable_vector_store', "Disable Vector Store?"), key="vs_toggle_key")
    
    default_k_vectorstore = 5
    if 'DEFAULT_TOP_K_VECTORSTORE' in st.secrets and isinstance(st.secrets['DEFAULT_TOP_K_VECTORSTORE'], dict):
        default_k_vectorstore = st.secrets['DEFAULT_TOP_K_VECTORSTORE'].get(username, 5)
    top_k_vectorstore = st.slider(lang_dict.get('top_k_vector_store', "Top-K for Vector Store"), 1, 10, default_k_vectorstore, disabled=disable_vector_store, key="vs_slider_key")
    
    rag_strategies_list = ('Basic Retrieval', 'Maximal Marginal Relevance', 'Fusion')
    default_rag_strat = 'Basic Retrieval'
    if 'DEFAULT_RAG_STRATEGY' in st.secrets and isinstance(st.secrets['DEFAULT_RAG_STRATEGY'], dict):
        default_rag_strat = st.secrets['DEFAULT_RAG_STRATEGY'].get(username, 'Basic Retrieval')
    strategy_idx_val = rag_strategies_list.index(default_rag_strat) if default_rag_strat in rag_strategies_list else 0
    strategy = st.selectbox(lang_dict.get('rag_strategy', "RAG Strategy"), rag_strategies_list, index=strategy_idx_val, help=lang_dict.get('rag_strategy_help',"Select RAG strategy"), disabled=disable_vector_store, key="rag_select_key")

    # Prompt Type and Custom Prompt
    custom_prompt_text_from_file = ""
    prompt_options_list = ('Short results', 'Extended results', 'Custom')
    custom_prompt_selected_idx = 0 # Default to "Short results"
    try:
        user_prompt_path = Path(f"./customizations/prompt/{username}.txt")
        default_prompt_path = Path("./customizations/prompt/default.txt")
        if user_prompt_path.is_file():
            custom_prompt_text_from_file = user_prompt_path.read_text(encoding='utf-8')
            custom_prompt_selected_idx = 2 # 'Custom'
        elif default_prompt_path.is_file():
            custom_prompt_text_from_file = default_prompt_path.read_text(encoding='utf-8')
            # Keep default index 0 for 'Short results', assuming default.txt is not a custom one by default
        else:
            custom_prompt_text_from_file = "Answer based on context: {context}\nQuestion: {question}" # Basic fallback
    except Exception as e:
        print(f"Error loading custom prompt text file: {e}")
        custom_prompt_text_from_file = "Error loading prompt."
        
    prompt_type = st.selectbox(lang_dict.get('system_prompt', "System Prompt"), prompt_options_list, index=custom_prompt_selected_idx, key="prompt_type_key")
    custom_prompt = st.text_area(lang_dict.get('custom_prompt', "Custom Prompt Text"), value=custom_prompt_text_from_file, help=lang_dict.get('custom_prompt_help', "Edit your custom prompt here"), disabled=(prompt_type != 'Custom'), key="custom_prompt_text_area")

    st.divider()
    # --- SECTIONS DE INGESTA COMENTADAS ---
    # # Include the upload form for new data to be Vectorized
    # # with st.sidebar.expander(lang_dict.get('load_context_expander', "Upload Files"), expanded=False): # If using expander
    # st.subheader(lang_dict.get('load_context_header', "Upload New Documents")) # Simple subheader
    # uploaded_files = st.file_uploader(lang_dict.get('load_context', "Upload TXT, PDF, CSV files"), type=['txt', 'pdf', 'csv'], accept_multiple_files=True, key="file_uploader_main")
    # upload_button_files = st.button(lang_dict.get('load_context_button', "Process Uploaded Files"), key="upload_files_button_main")
    # if upload_button_files and uploaded_files:
    #     if vectorstore: 
    #         with st.spinner(lang_dict.get('processing_files', "Processing files...")):
    #             vectorize_text(uploaded_files, vectorstore, lang_dict)
    #         st.success(lang_dict.get('files_processed_success', "Files processed successfully!"))
    #     else:
    #         st.error(lang_dict.get('vectorstore_not_ready_error', "Vector store not ready. Cannot process files."))

    # # Include the upload form for URLs be Vectorized
    # # with st.sidebar.expander(lang_dict.get('load_urls_expander', "Load from URLs"), expanded=False): # If using expander
    # st.subheader(lang_dict.get('load_urls_header', "Load Content from URLs")) # Simple subheader
    # urls_input_area = st.text_area(lang_dict.get('load_from_urls', "Enter URLs (comma-separated)"), help=lang_dict.get('load_from_urls_help', "Enter one or more URLs separated by commas."), key="urls_text_area_main")
    # upload_button_urls = st.button(lang_dict.get('load_from_urls_button', "Process URLs"), key="upload_urls_button_main")
    # if upload_button_urls and urls_input_area.strip():
    #     if vectorstore: 
    #         urls_list_to_process = [url.strip() for url in urls_input_area.split(',') if url.strip()]
    #         if urls_list_to_process:
    #             with st.spinner(lang_dict.get('processing_urls', "Processing URLs...")):
    #                 vectorize_url(urls_list_to_process, vectorstore, lang_dict)
    #             st.success(lang_dict.get('urls_processed_success', "URLs processed successfully!"))
    #         else:
    #             st.warning(lang_dict.get('no_valid_urls_warning', "No valid URLs entered."))
    #     else:
    #         st.error(lang_dict.get('vectorstore_not_ready_error', "Vector store not ready. Cannot process URLs."))
    
    # # Drop the vector data and start from scratch
    # # This is controlled by secrets.toml: [delete_option] username = "True" or "False"
    # # If you want to completely remove this button from the UI, comment out the entire 'if delete_option_is_true:' block
    # delete_option_is_true = False
    # if 'delete_option' in st.secrets and isinstance(st.secrets['delete_option'], dict) and username in st.secrets['delete_option']:
    #     delete_option_is_true = str(st.secrets.delete_option[username]).lower() == 'true'

    # if delete_option_is_true:
    #     st.divider()
    #     st.caption(lang_dict.get('delete_context', "Delete all vectorized content for this user."))
    #     if st.button(lang_dict.get('delete_context_button', "⚠️ Delete All Context"), key="delete_context_button_main"):
    #         if vectorstore and memory: 
    #             with st.spinner(lang_dict.get('deleting_context', "Deleting context...")):
    #                 vectorstore.clear() # Assuming vectorstore has a clear() method
    #                 memory.clear()
    #                 st.session_state.messages = [AIMessage(content=lang_dict.get('assistant_welcome', "Welcome! Context has been cleared."))]
    #             st.success("Context deleted successfully.")
    #             st.rerun() 
    #         else:
    #             st.error(lang_dict.get('cannot_delete_context_error' ,"Cannot delete context: Vectorstore or Memory not available."))
    # --- FIN DE BLOQUES COMENTADOS ---
    st.divider()
    # Camera input for image query
    st.subheader(lang_dict.get('image_query_header', "Query with Image"))
    picture = st.camera_input(lang_dict.get('take_picture', "Take a picture"), key="camera_input_main")
    if picture:
        if 'OPENAI_API_KEY' in st.secrets and st.secrets['OPENAI_API_KEY']:
            img_desc_response = describeImage(picture.getvalue(), language)
            if img_desc_response and img_desc_response.choices and img_desc_response.choices[0].message.content:
                picture_desc_text = img_desc_response.choices[0].message.content
                if not st.session_state.get("question_from_input"): # Check if text input already submitted question
                    question = picture_desc_text # Set as question if no text input
                    st.info(f"{lang_dict.get('image_desc_as_question', 'Using image description as question:')} {question[:100]}...")
            else:
                st.error(lang_dict.get('image_desc_fail_error', "Could
