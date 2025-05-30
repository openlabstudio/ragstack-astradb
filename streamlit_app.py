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

import openai

print("Started")
st.set_page_config(page_title='Your Enterprise Sidekick', page_icon='🚀')

# Get a unique session id for memory
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4() # Convert to string if it's not already

# Streaming call back handler for responses
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▌")

###############
### Globals ###
###############
# It's generally better to initialize these within a broader scope or pass as parameters,
# but following the original structure for now.
lang_dict = {}
language = "es_ES" # Default
rails_dict = {}
# session = None # This is a Streamlit object, not typically defined globally like this
embedding = None
vectorstore = None
chat_history = None
memory = None
disable_vector_store = False
strategy = 'Basic Retrieval'
prompt_type = 'Short results'
custom_prompt = ''


#################
### Functions ###
#################

# Close off the app using a password
def check_password():
    """Returns `True` if the user had a correct password."""

    def login_form():
        """Form with widgets to collect user information"""
        with st.form("credentials"):
            st.text_input('Username', key='username') # Original key
            st.text_input('Password', type='password', key='password') # Original key
            st.form_submit_button('Login', on_click=password_entered)

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        # Ensure secrets and keys exist before accessing
        if 'passwords' in st.secrets and \
           st.session_state.get('username') in st.secrets['passwords'] and \
           st.session_state.get('password') is not None and \
           hmac.compare_digest(st.session_state['password'], st.secrets.passwords[st.session_state['username']]):
            st.session_state['password_correct'] = True
            st.session_state.user = st.session_state['username']
            if 'password' in st.session_state: # Delete only if it exists
                 del st.session_state['password']  # Don't store the password.
        else:
            st.session_state['password_correct'] = False

    if st.session_state.get('password_correct', False):
        return True

    login_form()
    # Show error only if an attempt was made and it was incorrect
    if "password_correct" in st.session_state and not st.session_state.password_correct:
        st.error(lang_dict.get('login_error','😕 User not known or password incorrect')) # Use lang_dict.get
    return False

def logout():
    keys_to_delete = list(st.session_state.keys()) # Iterate over a copy
    for key in keys_to_delete:
        del st.session_state[key]
    st.cache_resource.clear()
    st.cache_data.clear()
    st.rerun()

# Function for Vectorizing uploaded data into Astra DB
def vectorize_text(uploaded_files):
    global vectorstore # Added to ensure it uses the global vectorstore if not passed
    global lang_dict   # Added for safety
    if not vectorstore:
        st.error(lang_dict.get('vectorstore_not_init_upload', "Vectorstore not initialized. Cannot process files."))
        return
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            
            # Write to temporary file
            with tempfile.TemporaryDirectory() as temp_dir: # Corrected usage
                file = uploaded_file
                print(f"""Processing: {file.name}""") # Use file.name
                temp_filepath = os.path.join(temp_dir, file.name) # Use temp_dir not temp_dir.name
                with open(temp_filepath, 'wb') as f:
                    f.write(file.getvalue())

                # Process TXT
                if uploaded_file.name.endswith('txt'):
                    with open(temp_filepath, 'r', encoding='utf-8') as f_txt: # Added encoding
                        file_content_list = [f_txt.read()]
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size = 1500,
                        chunk_overlap  = 100
                    )
                    texts = text_splitter.create_documents(file_content_list, [{'source': uploaded_file.name}] * len(file_content_list))
                    vectorstore.add_documents(texts)
                    st.info(f"{len(texts)} {lang_dict.get('load_text', 'text segments loaded')}")
                
                # Process PDF
                elif uploaded_file.name.endswith('pdf'): # Use elif for exclusive conditions
                    docs = []
                    loader = PyPDFLoader(temp_filepath) # Path should be string
                    docs.extend(loader.load())

                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size = 1500,
                        chunk_overlap  = 100
                    )
                    pages = text_splitter.split_documents(docs)
                    vectorstore.add_documents(pages)  
                    st.info(f"{len(pages)} {lang_dict.get('load_pdf', 'PDF pages/segments loaded')}")

                # Process CSV
                elif uploaded_file.name.endswith('csv'): # Use elif
                    docs = []
                    loader = CSVLoader(temp_filepath, encoding='utf-8') # Added encoding
                    docs.extend(loader.load())

                    vectorstore.add_documents(docs)
                    st.info(f"{len(docs)} {lang_dict.get('load_csv', 'CSV documents/rows loaded')}")


# Load data from URLs
def vectorize_url(urls_list): # Renamed parameter
    global vectorstore # Added
    global lang_dict   # Added
    if not vectorstore:
        st.error(lang_dict.get('vectorstore_not_init_url', "Vectorstore not initialized. Cannot process URLs."))
        return
    # Create the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1500,
        chunk_overlap  = 100
    )

    for url_item in urls_list: # Use the renamed parameter
        url = url_item.strip() # Clean each URL
        if not url: # Skip if URL is empty after stripping
            continue
        try:
            loader = WebBaseLoader(url) # Removed [url] as WebBaseLoader takes single URL or list
            docs = loader.load()    
            pages = text_splitter.split_documents(docs)
            print (f"Loading from URL: {pages}")
            vectorstore.add_documents(pages)  
            st.info(f"{len(pages)} {lang_dict.get('url_pages_loaded','loaded')}") # Used .get
        except Exception as e:
            st.info(f"{lang_dict.get('url_error', 'An error occurred')}:", e) # Used .get


# Define the prompt
def get_prompt(type_param, current_custom_prompt, current_language_code): # Renamed parameters
    template = ''
    # Ensure language is a string for f-string, use global `language` if current_language_code is not passed properly
    lang_code_for_fstring = current_language_code if isinstance(current_language_code, str) else language

    base_template_text = f"""Use the following context to answer the question:
{{context}}

Use the following chat history to answer the question:
{{chat_history}}

Question:
{{question}}

Answer in {lang_code_for_fstring}:"""

    if type_param == 'Extended results':
        print ("Prompt type: Extended results")
        template = f"""You're a helpful AI assistant tasked to answer the user's questions.
You're friendly and you answer extensively with multiple sentences. You prefer to use bulletpoints to summarize.
If the question states the name of the user, just say 'Thanks, I'll use this information going forward'.
If you don't know the answer, just say 'I do not know the answer'.
{base_template_text}"""

    elif type_param == 'Short results': # Use elif
        print ("Prompt type: Short results")
        template = f"""You're a helpful AI assistant tasked to answer the user's questions.
You answer in an exceptionally brief way.
If the question states the name of the user, just say 'Thanks, I'll use this information going forward'.
If you don't know the answer, just say 'I do not know the answer'.
{base_template_text}"""

    elif type_param == 'Custom': # Use elif
        print ("Prompt type: Custom")
        template = str(current_custom_prompt) if current_custom_prompt else base_template_text # Ensure string and provide fallback
    else: # Default case
        template = base_template_text


    return ChatPromptTemplate.from_messages([("system", template)])

# Get the OpenAI Chat Model
# Original function name was load_model, changed to load_model_cached for clarity with @st.cache_resource
@st.cache_resource(show_spinner=True) # Simplified spinner
def load_model_cached(): # Renamed to avoid conflict
    print(f"""load_model_cached""")
    # Get the OpenAI Chat Model
    return ChatOpenAI(
        temperature=0.3,
        model='gpt-4-1106-preview', # Consider changing to gpt-4o or newer
        streaming=True,
        verbose=True # Often set to False for production
    )

# Get the Retriever
def load_retriever(current_vectorstore, top_k_vs_param): # Renamed parameters
    print(f"""load_retriever with top_k_vectorstore='{top_k_vs_param}'""")
    # Get the Retriever from the Vectorstore
    if not current_vectorstore: # Add check
        return None
    return current_vectorstore.as_retriever(
        search_kwargs={"k": top_k_vs_param}
    )

@st.cache_resource(show_spinner=True) # Simplified spinner
def load_memory_cached(_chat_history, top_k_hist_param): # Renamed parameters
    print(f"""load_memory_cached with top-k={top_k_hist_param}""")
    if not _chat_history: # Add check
        return None
    return ConversationBufferWindowMemory(
        chat_memory=_chat_history, # Use the passed parameter
        return_messages=True,
        k=top_k_hist_param,
        memory_key="chat_history",
        input_key="question",
        output_key='answer',
    )

def generate_queries_chain(current_model, current_language): # Renamed and added parameters
    if not current_model: return None # Add check
    prompt_text = f"""You are a helpful assistant that generates multiple search queries based on a single input query in language {current_language}.
Generate multiple search queries related to: {{original_query}}
OUTPUT (4 queries):"""

    return ChatPromptTemplate.from_messages([("system", prompt_text)]) | current_model | StrOutputParser() | (lambda x: x.split("\n"))


def reciprocal_rank_fusion(results: list[list], k_rrf=60): # Renamed parameter
    from langchain.load import dumps, loads # Local import

    fused_scores = {}
    for docs_list_item in results: # Renamed variable for clarity
        if not isinstance(docs_list_item, list): continue # Skip if not a list
        for rank, doc in enumerate(docs_list_item):
            try:
                doc_str = dumps(doc) # This expects Langchain Document or similar serializable object
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                # previous_score = fused_scores[doc_str] # This variable was unused
                fused_scores[doc_str] += 1 / (rank + k_rrf)
            except Exception as e:
                print(f"Could not dump/process doc in RRF: {e}") # Log error
                continue # Skip this doc

    reranked_results = []
    for doc_s, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True):
        try:
            reranked_results.append((loads(doc_s), score))
        except Exception as e:
            print(f"Could not load/process doc_s in RRF: {e}")
            continue
    return reranked_results


# Describe the image based on OpenAI
def describeImage(image_bin_param, lang_param): # Renamed parameters
    print ("describeImage")
    # Ensure openai.api_key is set, typically from secrets or environment
    if not (st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        st.error("OpenAI API Key not configured.")
        return None
    
    # Initialize client inside the function or ensure it's globally available and initialized
    client = openai.OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY")))

    image_base64 = base64.b64encode(image_bin_param).decode()
    try:
        response = client.chat.completions.create( # Use the client instance
            model="gpt-4-vision-preview",
            messages=[
                {
                "role": "user",
                "content": [
                    #{"type": "text", "text": "Describe the image in detail"},
                    {"type": "text", "text": f"Provide a search text for the main topic of the image writen in {lang_param}"},
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}",
                    },
                    },
                ],
                }
            ],
            max_tokens=4096,  # default max tokens is low so set higher
        )
        print (f"describeImage result: {response}")
        return response
    except Exception as e:
        st.error(f"Error in describeImage: {e}")
        return None


##################
### Data Cache ###
##################

# Cache localized strings
@st.cache_data()
def load_localization(locale_param): # Renamed parameter
    print(f"load_localization for: {locale_param}")
    try:
        df = pd.read_csv("./customizations/localization.csv", encoding='utf-8') # Added encoding
        df_lang = df.query(f"locale == '{locale_param}'")
        if df_lang.empty and locale_param != "en_US": # Fallback to en_US
            print(f"Locale {locale_param} not found, attempting en_US.")
            df_lang = df.query("locale == 'en_US'")
        if df_lang.empty: # If en_US also not found
            print("Warning: Neither specified locale nor en_US found in localization.csv. Using minimal defaults.")
            return {"assistant_welcome": "Welcome!", "logout_caption": "Logged in as", "logout_button": "Logout", "assistant_question": "Your question...", "login_error": "Login failed."}
        # Create and return a dictionary of key/values.
        lang_dict_loaded = pd.Series(df_lang.value.values,index=df_lang.key).to_dict() # Renamed
        return lang_dict_loaded
    except FileNotFoundError:
        print("ERROR: localization.csv not found. Using minimal defaults.")
        return {"assistant_welcome": "Welcome! (localization.csv missing)", "logout_caption": "Logged in as", "logout_button": "Logout", "assistant_question": "Your question...", "login_error": "Login failed."}
    except Exception as e: # Catch other potential errors during CSV processing
        print(f"Error loading localization.csv: {e}. Using minimal defaults.")
        return {"assistant_welcome": "Welcome! (localization error)", "logout_caption": "Logged in as", "logout_button": "Logout", "assistant_question": "Your question...", "login_error": "Login failed."}


# Cache localized strings
@st.cache_data()
def load_rails(username_param): # Renamed parameter
    print(f"load_rails for {username_param}")
    try:
        df = pd.read_csv("./customizations/rails.csv", encoding='utf-8') # Added encoding
        df = df.query(f"username == '{username_param}'")
        if df.empty:
            return {} # No rails for this user
        # Create and return a dictionary of key/values.
        rails_dict_loaded = pd.Series(df.value.values,index=df.key).to_dict() # Renamed
        return rails_dict_loaded
    except FileNotFoundError:
        print("Warning: rails.csv not found. No rails loaded.")
        return {}
    except Exception as e: # Catch other potential errors
        print(f"Error loading rails for {username_param}: {e}")
        return {}


#############
### Login ###
#############

# Initialize lang_dict with minimal defaults BEFORE login attempt so st.error in check_password can use it
lang_dict = {"login_error": '😕 User not known or password incorrect'}
if not check_password():
    st.stop()  # Do not continue if check_password is not True.

username = st.session_state.user # Set by check_password()
# Load full lang_dict after successful login
language = st.secrets.get("languages", {}).get(username, "es_ES") # Default to Spanish, use .get for safety
lang_dict = load_localization(language) # Now load the full dictionary

#######################
### Resources Cache ###
#######################

# Cache OpenAI Embedding for future runs
@st.cache_resource(show_spinner=lang_dict.get('load_embedding', "Loading Embeddings...")) # Use .get
def load_embedding_rc(): # Renamed to avoid conflict with global `embedding`
    print("load_embedding_rc")
    # Get the OpenAI Embedding
    try:
        # Ensure API key is fetched correctly from secrets
        openai_api_key_val = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY"))
        if not openai_api_key_val:
            st.error("OpenAI API Key is not configured in st.secrets or environment variables.")
            return None
        return OpenAIEmbeddings(openai_api_key=openai_api_key_val)
    except Exception as e:
        st.error(f"Error loading OpenAI Embeddings: {e}")
        return None


# Cache Vector Store for future runs
@st.cache_resource(show_spinner=lang_dict.get('load_vectorstore', "Loading Vectorstore...")) # Use .get
def load_vectorstore_rc(username_param, embedding_instance): # Renamed, pass embedding_instance
    print(f"load_vectorstore_rc for {username_param}")
    if not embedding_instance: # Check if embedding was loaded
        st.error("Embedding model not loaded. Cannot initialize vector store.")
        return None
    try:
        # Get the vectorstore store from Astra DB
        # Ensure secrets are used correctly
        astra_token_val = st.secrets.get("ASTRA_TOKEN")
        astra_endpoint_val = st.secrets.get("ASTRA_ENDPOINT", os.environ.get("ASTRA_ENDPOINT")) # Allow fallback to ENV for endpoint
        astra_keyspace_val = st.secrets.get("ASTRA_KEYSPACE") # Optional keyspace from secrets

        if not astra_token_val or not astra_endpoint_val:
            st.error("Astra DB Token or API Endpoint not found in st.secrets.")
            return None

        return AstraDB(
            embedding=embedding_instance, # Use passed embedding_instance
            collection_name=f"vector_context_{username_param}",
            token=astra_token_val,
            api_endpoint=astra_endpoint_val,
            namespace=astra_keyspace_val # Pass namespace if using specific keyspace
        )
    except Exception as e:
        st.error(f"Error initializing AstraDB vector store: {e}")
        return None

# Cache Chat History for future runs
@st.cache_resource(show_spinner=lang_dict.get('load_message_history', "Loading Chat History...")) # Use .get
def load_chat_history_rc(username_param, session_id_param): # Renamed
    print(f"load_chat_history_rc for {username_param}_{session_id_param}")
    try:
        astra_token_val = st.secrets.get("ASTRA_TOKEN")
        astra_endpoint_val = st.secrets.get("ASTRA_ENDPOINT", os.environ.get("ASTRA_ENDPOINT"))
        astra_keyspace_val = st.secrets.get("ASTRA_KEYSPACE")

        if not astra_token_val or not astra_endpoint_val:
            st.error("Astra DB Token or API Endpoint not found for chat history.")
            return None

        return AstraDBChatMessageHistory(
            session_id=f"{username_param}_{str(session_id_param)}", # Ensure session_id is string
            api_endpoint=astra_endpoint_val,
            token=astra_token_val,
            keyspace_name=astra_keyspace_val # Pass keyspace_name
        )
    except Exception as e:
        st.error(f"Error initializing AstraDB chat history: {e}")
        return None

# Load resources after login and lang_dict is fully available
embedding = load_embedding_rc() # Call renamed cached function
if embedding:
    vectorstore = load_vectorstore_rc(username, embedding) # Pass username and loaded embedding
if vectorstore: # Only load chat history if vectorstore was successful (implies Astra creds are ok)
    chat_history = load_chat_history_rc(username, st.session_state.session_id)


#####################
### Session state ###
#####################

# Start with empty messages, stored in session state
if 'messages' not in st.session_state:
    st.session_state.messages = [AIMessage(content=lang_dict.get('assistant_welcome', "Welcome!"))] # Used .get

############
### Main ###
############

# Show a custom welcome text or the default text
try:
    # Ensure username is valid for path construction
    safe_username = "".join(c if c.isalnum() else "_" for c in username) # Sanitize username for path
    welcome_file = Path(f"./customizations/welcome/{safe_username}.md")
    if welcome_file.is_file():
        st.markdown(welcome_file.read_text(encoding='utf-8'))
    else: # Fallback to default if user-specific not found
        st.markdown(Path('./customizations/welcome/default.md').read_text(encoding='utf-8'))
except FileNotFoundError: # Specifically catch if default.md is also missing
    st.markdown(lang_dict.get('assistant_welcome', "Welcome!")) # Fallback if no md files
except Exception as e: # Catch other errors
    print(f"Error loading welcome.md: {e}")
    st.markdown(lang_dict.get('assistant_welcome', "Welcome!"))


# Show a custom logo (svg or png) or the DataStax logo
with st.sidebar:
    try:
        safe_username_logo = "".join(c if c.isalnum() else "_" for c in username) # Sanitize
        user_logo_svg_path = Path(f"./customizations/logo/{safe_username_logo}.svg")
        user_logo_png_path = Path(f"./customizations/logo/{safe_username_logo}.png")
        default_logo_path = Path('./customizations/logo/default.svg')
        
        logo_to_show = None
        if user_logo_svg_path.is_file():
            logo_to_show = str(user_logo_svg_path)
        elif user_logo_png_path.is_file():
            logo_to_show = str(user_logo_png_path)
        elif default_logo_path.is_file():
            logo_to_show = str(default_logo_path)

        if logo_to_show:
            st.image(logo_to_show, use_column_width="always")
        else:
            st.text("Logo") # Fallback
        st.text('') # For spacing
    except Exception as e:
        print(f"Error loading logo: {e}")
        st.text("Logo") # Fallback

# Logout button
with st.sidebar:
    st.markdown(f"""{lang_dict.get('logout_caption', "Logged in as")} :orange[{username}]""")
    logout_button = st.button(lang_dict.get('logout_button', "Logout"))
    if logout_button:
        logout()

with st.sidebar:
    st.divider()

# Initialize resources that might depend on prior initializations
# model is loaded here as it doesn't depend on username directly like others
model = load_model_cached() # Original 'load_model' renamed to 'load_model_cached' for cache consistency

# Options panel
with st.sidebar:
    # Chat history settings
    rails_dict = load_rails(username) # Load rails
    disable_chat_history = st.toggle(lang_dict.get('disable_chat_history',"Disable Chat History?"), key="ch_toggle") # Added .get
    
    # Get default from secrets or use a hardcoded default
    default_top_k_history = st.secrets.get("DEFAULT_TOP_K_HISTORY", {}).get(username, 5)
    top_k_history = st.slider(lang_dict.get('k_chat_history',"K for Chat History"), 1, 50, default_top_k_history, disabled=disable_chat_history, key="ch_slider") # Added .get
    
    # Load memory after top_k_history is set
    if chat_history: # Ensure chat_history object exists
        memory = load_memory_cached(chat_history, top_k_history if not disable_chat_history else 0)
    else:
        memory = None # Set memory to None if chat_history failed to load

    if memory: # Only show button if memory is successfully loaded
        delete_history = st.button(lang_dict.get('delete_chat_history_button', "Delete Chat History"), disabled=disable_chat_history, key="del_hist_btn") # Added .get
        if delete_history:
            with st.spinner(lang_dict.get('deleting_chat_history', "Deleting...")): # Added .get
                memory.clear()
            st.session_state.messages = [AIMessage(content=lang_dict.get('assistant_welcome', "Welcome!"))] # Reset messages
            st.rerun() # Rerun to clear displayed messages

    # Vector store settings
    disable_vector_store = st.toggle(lang_dict.get('disable_vector_store', "Disable Vector Store?"), key="vs_toggle") # Added .get
    default_top_k_vectorstore = st.secrets.get("DEFAULT_TOP_K_VECTORSTORE", {}).get(username, 5)
    top_k_vectorstore = st.slider(lang_dict.get('top_k_vector_store', "Top-K for Vector Store"), 1, 50, default_top_k_vectorstore, disabled=disable_vector_store, key="vs_slider") # Added .get
    
    rag_strategy_options = ('Basic Retrieval', 'Maximal Marginal Relevance', 'Fusion')
    default_strategy_val = st.secrets.get("DEFAULT_RAG_STRATEGY", {}).get(username, 'Basic Retrieval')
    strategy_idx = rag_strategy_options.index(default_strategy_val) if default_strategy_val in rag_strategy_options else 0
    strategy = st.selectbox(lang_dict.get('rag_strategy', "RAG Strategy"), rag_strategy_options, index=strategy_idx, help=lang_dict.get('rag_strategy_help', "Help"), disabled=disable_vector_store, key="rag_selectbox") # Added .get

    custom_prompt_text_val = '' # Renamed
    custom_prompt_idx_val = 0 # Renamed
    try:
        safe_username_prompt = "".join(c if c.isalnum() else "_" for c in username) # Sanitize
        custom_prompt_file = Path(f"""./customizations/prompt/{safe_username_prompt}.txt""")
        default_prompt_file = Path(f"""./customizations/prompt/default.txt""")
        if custom_prompt_file.is_file():
            custom_prompt_text_val = custom_prompt_file.read_text(encoding='utf-8') # Added encoding
            custom_prompt_idx_val = 2 # Index for 'Custom'
        elif default_prompt_file.is_file():
            custom_prompt_text_val = default_prompt_file.read_text(encoding='utf-8') # Added encoding
            custom_prompt_idx_val = 0 # Or other appropriate default index
        else: # Fallback if no prompt files
            custom_prompt_text_val = "Fallback prompt if files are missing."
    except Exception as e:
        print(f"Error loading prompt file: {e}")
        custom_prompt_text_val = "Error loading prompt."

    prompt_type_options = ('Short results', 'Extended results', 'Custom')
    prompt_type = st.selectbox(lang_dict.get('system_prompt', "System Prompt"), prompt_type_options, index=custom_prompt_idx_val, key="prompt_select") # Added .get
    custom_prompt = st.text_area(lang_dict.get('custom_prompt', "Custom Prompt Text"), custom_prompt_text_val, help=lang_dict.get('custom_prompt_help', "Help"), disabled=(prompt_type != 'Custom'), key="custom_prompt_text") # Added .get
    print(f"""Sidebar Config: DisableVS={disable_vector_store}, K_Hist={top_k_history}, K_VS={top_k_vectorstore}, Strategy={strategy}, PromptType={prompt_type}""")

with st.sidebar:
    st.divider()

# --- BLOQUES DE INGESTA COMENTADOS ---
# # Include the upload form for new data to be Vectorized
# with st.sidebar:
#     # st.subheader(lang_dict.get('upload_header', "Upload Documents")) # Example using .get
#     uploaded_files = st.file_uploader(lang_dict.get('load_context', "Upload Files (TXT, PDF, CSV)"), type=['txt', 'pdf', 'csv'], accept_multiple_files=True, key="fileuploader")
#     upload_files_button = st.button(lang_dict.get('load_context_button', "Process Files"), key="uploadbutton") # Renamed variable
#     if upload_files_button and uploaded_files:
#         vectorize_text(uploaded_files, vectorstore, lang_dict) # Pass dependencies

# # Include the upload form for URLs be Vectorized
# with st.sidebar:
#     # st.subheader(lang_dict.get('url_upload_header', "Load from URLs")) # Example using .get
#     urls_text_area = st.text_area(lang_dict.get('load_from_urls', "Enter URLs (comma-separated)"), help=lang_dict.get('load_from_urls_help', "One URL per line or comma-separated"), key="urltextarea") # Renamed variable
#     urls_to_process = [url.strip() for url in urls_text_area.split(',') if url.strip()] # Process here
#     upload_urls_button = st.button(lang_dict.get('load_from_urls_button', "Process URLs"), key="uploadurlbutton") # Renamed variable
#     if upload_urls_button and urls_to_process:
#         vectorize_url(urls_to_process, vectorstore, lang_dict) # Pass dependencies

# # Drop the vector data and start from scratch
# # This is controlled by secrets.toml: [delete_option] username = "True" or "False"
# delete_option_setting = str(st.secrets.get("delete_option", {}).get(username, "False")).lower() == 'true' # Safer get
# if delete_option_setting:
#     with st.sidebar:
#         st.caption(lang_dict.get('delete_context', "Delete all context for this user.")) # Use .get
#         submitted_delete_btn = st.button(lang_dict.get('delete_context_button', "⚠️ Delete Context"), key="deletecontextbutton") # Renamed variable
#         if submitted_delete_btn:
#             if vectorstore and memory: # Check if resources are available
#                 with st.spinner(lang_dict.get('deleting_context', "Deleting context...")): # Use .get
#                     try:
#                         vectorstore.clear() # Assuming this method exists and works for AstraDB via Langchain
#                         memory.clear()
#                         st.session_state.messages = [AIMessage(content=lang_dict.get('assistant_welcome', "Welcome! Context cleared."))] # Use .get
#                         st.success("Context deleted.")
#                     except Exception as e:
#                         st.error(f"Error clearing context: {e}")
#                 st.rerun()
#             else:
#                 st.warning("Vectorstore or memory not available. Cannot delete context.")
# --- FIN DE BLOQUES COMENTADOS ---


with st.sidebar:
    st.divider()

# Draw rails
with st.sidebar:
    st.subheader(lang_dict.get('rails_1', "Suggestions")) # Use .get
    st.caption(lang_dict.get('rails_2', "Try asking:")) # Use .get
    if rails_dict:
        for i in sorted(rails_dict.keys()): # Sort for consistent order
            st.markdown(f"{i}. {rails_dict[i]}")
    else: # Handle empty rails_dict
        st.markdown(lang_dict.get('no_rails_prompt', "No specific suggestions for you.")) # Use .get


# Draw all messages, both user and agent so far (every time the app reruns)
for message_item in st.session_state.messages: # Renamed variable
    if hasattr(message_item, 'type') and hasattr(message_item, 'content'):
        st.chat_message(message_item.type).markdown(message_item.content)
    # Fallback for dicts (though Langchain messages are objects)
    elif isinstance(message_item, dict) and "role" in message_item and "content" in message_item:
        st.chat_message(message_item["role"]).markdown(message_item["content"])


# Now get a prompt from a user
# Use a different key for chat_input to avoid conflict with other inputs
question_text = st.chat_input(lang_dict.get('assistant_question', "Ask your question here..."), key="user_main_question_input") # Renamed and used .get

st.session_state.question_from_text_input = bool(question_text) # Flag if question came from text input

# --- BLOQUE DE CÁMARA COMENTADO ---
# with st.sidebar:
#     st.divider()
#     # st.subheader(lang_dict.get('image_query_header', "Query with Image")) # Use .get
#     # picture_data = st.camera_input(lang_dict.get('take_picture', "Take a picture"), key="main_camera_input") # Renamed
#     # if picture_data:
#     #     openai_api_key_val = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY"))
#     #     if openai_api_key_val: # Check if key is actually available
#     #         # Ensure openai client is initialized if not done globally
#     #         # openai.api_key = openai_api_key_val # This might be needed if not using instance
            
#     #         image_description_response = describeImage(picture_data.getvalue(), language) # Pass language
#     #         if image_description_response and image_description_response.choices and image_description_response.choices[0].message.content:
#     #             image_description_text = image_description_response.choices[0].message.content # Renamed
#     #             # Only use if no text question was submitted simultaneously
#     #             if not question_text and not st.session_state.question_from_text_input: 
#     #                 question_text = image_description_text # This will be picked up below
#     #                 st.info(f"{lang_dict.get('image_desc_as_question', 'Using image description as question:')} {question_text[:100]}...")
#     #         else:
#     #             st.error(lang_dict.get('image_desc_fail_error', "Could not get description from image."))
#     #     else:
#     #         st.error(lang_dict.get('openai_key_missing_camera_error', "OpenAI API Key not configured. Camera feature disabled."))
# --- FIN DE BLOQUE DE CÁMARA COMENTADO ---


if question_text: # If there is a question (from chat_input or potentially from camera if that logic was active)
    print(f"Got question: {question_text}")
            
    st.session_state.messages.append(HumanMessage(content=question_text))
    with st.chat_message('human'):
        st.markdown(question_text)

    with st.chat_message('assistant'):
        response_placeholder = st.empty()
        
        # Initialize to defaults
        retriever_instance = None # Renamed
        relevant_docs_list = [] # Renamed

        if vectorstore and not disable_vector_store:
            retriever_instance = load_retriever(vectorstore, top_k_vectorstore) # Original function name
        
        if retriever_instance: # Check if retriever was successfully created
            if strategy == 'Basic Retrieval':
                relevant_docs_list = retriever_instance.get_relevant_documents(query=question_text) # Pass k from slider
            elif strategy == 'Maximal Marginal Relevance':
                # MMR search is usually a method of the vectorstore itself, not the retriever
                if vectorstore: # Ensure vectorstore is available
                    relevant_docs_list = vectorstore.max_marginal_relevance_search(query=question_text, k=top_k_vectorstore, fetch_k=20) # fetch_k is often > k
            elif strategy == 'Fusion':
                if model and retriever_instance: # Model for query generation, retriever for subsequent searches
                    query_gen_chain = generate_queries_chain(model, language) # Original function name
                    if query_gen_chain:
                        fusion_queries_list = query_gen_chain.invoke({"original_query": question_text})
                        print(f"""Fusion queries: {fusion_queries_list}""")

                        fusion_queries_display_text = f"""*{lang_dict.get('using_fusion_queries', "Using fusion queries:")}*\n""" # Renamed
                        for i, fq_item in enumerate(fusion_queries_list): # Renamed
                            fusion_queries_display_text += f"""\n{i+1}. :orange[{fq_item}]"""
                        
                        # Display fusion queries as an intermediate step (added to messages list later)
                        # response_placeholder.markdown(fusion_queries_display_text) 
                        st.session_state.messages.append(AIMessage(content=fusion_queries_display_text))
                        # Rerun to show this intermediate message before the final answer streams
                        # This might make the UX a bit jumpy, consider if this display is essential here

                        retrieved_docs_for_fusion_lists = retriever_instance.map().invoke(fusion_queries_list) # list of lists
                        fused_docs_with_scores_list = reciprocal_rank_fusion(retrieved_docs_for_fusion_lists) # Renamed
                        relevant_docs_list = [doc_tuple[0] for doc_tuple in fused_docs_with_scores_list][:top_k_vectorstore] # Get top_k
                        print(f"""Fusion results (docs): {relevant_docs_list}""")
                    else: # Fallback if query_gen_chain failed
                        relevant_docs_list = retriever_instance.get_relevant_documents(query=question_text)
                else: # Fallback if model not available for query generation
                    relevant_docs_list = retriever_instance.get_relevant_documents(query=question_text)
        
        # Prepare for LLM
        final_llm_response_content = '' # Renamed
        chat_history_for_llm_invoke = {"chat_history": []} # Renamed
        if memory: # Check if memory exists
            chat_history_for_llm_invoke = memory.load_memory_variables({})
        print(f"Using memory: {chat_history_for_llm_invoke}")

        if model: # Check if LLM model exists
            current_prompt_obj = get_prompt(prompt_type, custom_prompt, language) # Renamed var
            
            inputs_map_for_chain = RunnableMap({ # Renamed var
                'context': lambda x: x['context'],
                'chat_history': lambda x: x['chat_history'], # This should be list of Message objects
                'question': lambda x: x['question']
            })
            print(f"Using inputs map: {inputs_map_for_chain}")

            full_rag_chain = inputs_map_for_chain | current_prompt_obj | model # Renamed var
            print(f"Using full RAG chain: {full_rag_chain}")

            try:
                llm_response_object = full_rag_chain.invoke( # Renamed var
                    {'question': question_text, 'chat_history': chat_history_for_llm_invoke.get('chat_history', []), 'context': relevant_docs_list}, 
                    config={'callbacks': [StreamHandler(response_placeholder)]}
                )
                print(f"LLM Response object: {llm_response_object}")
                final_llm_response_content = llm_response_object.content if hasattr(llm_response_object, 'content') else str(llm_response_object)
            except Exception as e:
                print(f"Error invoking RAG chain: {e}")
                st.error(f"Error during RAG chain invocation: {e}")
                final_llm_response_content = lang_dict.get("error_rag_chain", "Sorry, an error occurred while generating the response.")

        else: # Model not loaded
            final_llm_response_content = lang_dict.get("model_not_loaded_error", "AI Model is not available at the moment.")
            st.warning(final_llm_response_content)


        if memory: # Check if memory exists
            memory.save_context({'question': question_text}, {'answer': final_llm_response_content})

        # Append sources and history info to the final displayed message
        message_to_display_final = final_llm_response_content # Renamed

        if not disable_vector_store and relevant_docs_list:
            message_to_display_final += f"\n\n*{lang_dict.get('sources_used', 'Sources:')}*"
            displayed_sources_list_final = [] # Renamed
            for doc_item_final in relevant_docs_list: # Renamed var
                source_from_meta = doc_item_final.metadata.get('source', 'Unknown Source') if hasattr(doc_item_final, 'metadata') and doc_item_final.metadata else 'Unknown Source'
                source_basename_final = os.path.basename(os.path.normpath(source_from_meta)) # Renamed
                if source_basename_final not in displayed_sources_list_final:
                    message_to_display_final += f"\n📙 :orange[{source_basename_final}]"
                    displayed_sources_list_final.append(source_basename_final)
        elif not disable_vector_store: # Vector store enabled but no relevant_docs found
             message_to_display_final += f"\n\n*{lang_dict.get('no_docs_found_query', 'No specific documents found for this query in the knowledge base.')}*"


        if not disable_chat_history and memory and chat_history_for_llm_invoke.get('chat_history', []): # Check memory also
            num_history_pairs_final = len(chat_history_for_llm_invoke['chat_history']) // 2 # Renamed
            message_to_display_final += f"\n\n*{lang_dict.get('chat_history_info_display', 'Chat history turns considered')}: ({num_history_pairs_final}/{top_k_history})*" # Used .get
        
        response_placeholder.markdown(message_to_display_final) # Update placeholder with final content
        st.session_state.messages.append(AIMessage(content=message_to_display_final))
    
    # Reset question from session state if it was from camera (now a moot point as camera is commented)
    # if "question" in st.session_state and not st.session_state.get("question_from_text_input", False):
    # del st.session_state.question 
    st.session_state.question_from_text_input = False # Reset flag


with st.sidebar:
    st.divider()
    st.caption(lang_dict.get("app_version_display", "v231227.01_no_user_ingest")) # Used .get, updated version
