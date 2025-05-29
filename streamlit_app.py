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

print("Started") # For Replit console debugging
st.set_page_config(page_title='Your Enterprise Sidekick', page_icon='🚀')

# Get a unique session id for memory
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4()

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

global lang_dict
global language
global rails_dict
# session is a Streamlit object, no need to global declare here
global embedding
global vectorstore
global chat_history
global memory

# RAG options
global disable_vector_store
global strategy
global prompt_type
global custom_prompt

#################
### Functions ###
#################

# Close off the app using a password
def check_password():
    """Returns `True` if the user had a correct password."""

    def login_form():
        """Form with widgets to collect user information"""
        with st.form("credentials"):
            st.text_input('Username', key='username')
            st.text_input('Password', type='password', key='password')
            st.form_submit_button('Login', on_click=password_entered)

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state['username'] in st.secrets['passwords'] and hmac.compare_digest(st.session_state['password'], st.secrets.passwords[st.session_state['username']]):
            st.session_state['password_correct'] = True
            st.session_state.user = st.session_state['username']
            del st.session_state['password']  # Don't store the password.
        else:
            st.session_state['password_correct'] = False

    # Return True if the username + password is validated.
    if st.session_state.get('password_correct', False):
        return True

    # Show inputs for username + password.
    login_form()
    if "password_correct" in st.session_state:
        st.error('😕 User not known or password incorrect')
    return False

def logout():
    for key in list(st.session_state.keys()): # Use list() for safe iteration while deleting
        del st.session_state[key]
    st.cache_resource.clear()
    st.cache_data.clear()
    st.rerun()

# Function for Vectorizing uploaded data into Astra DB (Mantendremos estas funciones por si las usáis en modo admin internamente)
def vectorize_text(uploaded_files):
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            
            # Write to temporary file
            temp_dir = tempfile.TemporaryDirectory()
            file = uploaded_file
            # print(f"""Processing: {file}""") # Replit console
            st.session_state.debug_messages.append(f"Processing file: {file.name}")


            temp_filepath = os.path.join(temp_dir.name, file.name)
            with open(temp_filepath, 'wb') as f:
                f.write(file.getvalue())

            # Process TXT
            if uploaded_file.name.endswith('txt'):
                # It's better to load it as a Document object for consistency
                with open(temp_filepath, 'r', encoding='utf-8') as f_txt:
                    file_content = [f_txt.read()]
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size = 1500,
                    chunk_overlap  = 100
                )
                texts = text_splitter.create_documents(file_content, [{'source': uploaded_file.name}])
                vectorstore.add_documents(texts)
                st.info(f"{len(texts)} {lang_dict.get('load_text', 'text segments loaded')}") # Use .get for safety
            
            # Process PDF
            if uploaded_file.name.endswith('pdf'):
                docs = []
                loader = PyPDFLoader(temp_filepath)
                docs.extend(loader.load())

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size = 1500,
                    chunk_overlap  = 100
                )
                pages = text_splitter.split_documents(docs)
                vectorstore.add_documents(pages)  
                st.info(f"{len(pages)} {lang_dict.get('load_pdf', 'PDF pages/segments loaded')}")

            # Process CSV
            if uploaded_file.name.endswith('csv'):
                docs = []
                loader = CSVLoader(temp_filepath)
                docs.extend(loader.load())
                # For CSVs, you might want to split differently or ensure content is text-rich
                vectorstore.add_documents(docs)
                st.info(f"{len(docs)} {lang_dict.get('load_csv', 'CSV documents/rows loaded')}")
            
            temp_dir.cleanup() # Clean up temporary directory

# Load data from URLs (Mantendremos estas funciones por si las usáis en modo admin internamente)
def vectorize_url(urls):
    # Create the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1500,
        chunk_overlap  = 100
    )

    for url_item in urls:
        url = url_item.strip() # Clean whitespace
        if not url:
            continue
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()    
            pages = text_splitter.split_documents(docs)
            # print (f"Loading from URL: {pages}") # Replit console
            st.session_state.debug_messages.append(f"Loading from URL ({len(pages)} pages): {url}")
            vectorstore.add_documents(pages)  
            st.info(f"{len(pages)} {lang_dict.get('url_pages_loaded', 'pages loaded from URL')}")
        except Exception as e:
            st.error(f"{lang_dict.get('url_error', 'Error loading from URL')} {url}: {e}")
            st.session_state.debug_messages.append(f"Error loading URL {url}: {e}")


# Define the prompt
def get_prompt(type):
    template = ''
    current_language = st.session_state.get("language", "es_ES") # Get current language from session state

    if type == 'Extended results':
        # print ("Prompt type: Extended results") # Replit console
        template = f"""You're a helpful AI assistant tasked to answer the user's questions.
You're friendly and you answer extensively with multiple sentences. You prefer to use bulletpoints to summarize.
If the question states the name of the user, just say 'Thanks, I'll use this information going forward'.
If you don't know the answer, just say 'I do not know the answer'.

Use the following context to answer the question:
{{context}}

Use the following chat history to answer the question:
{{chat_history}}

Question:
{{question}}

Answer in {current_language}:""" # Use dynamic language

    if type == 'Short results':
        # print ("Prompt type: Short results") # Replit console
        template = f"""You're a helpful AI assistant tasked to answer the user's questions.
You answer in an exceptionally brief way.
If the question states the name of the user, just say 'Thanks, I'll use this information going forward'.
If you don't know the answer, just say 'I do not know the answer'.

Use the following context to answer the question:
{{context}}

Use the following chat history to answer the question:
{{chat_history}}

Question:
{{question}}

Answer in {current_language}:""" # Use dynamic language

    if type == 'Custom':
        # print ("Prompt type: Custom") # Replit console
        template = custom_prompt # Assumes custom_prompt is globally available or passed if needed

    return ChatPromptTemplate.from_messages([("system", template)])

# Get the OpenAI Chat Model
@st.cache_resource() # Cache the model loading
def load_model():
    # print(f"""load_model""") # Replit console
    # Get the OpenAI Chat Model
    return ChatOpenAI(
        temperature=0.3,
        model='gpt-4o', # Using a more recent model, ensure it's available in your OpenAI plan
        streaming=True,
        verbose=False # Usually False for production/demos unless debugging Langchain calls
    )

# Get the Retriever
# We don't cache the retriever directly if top_k_vectorstore can change dynamically
def load_retriever(current_vectorstore, top_k_vectorstore):
    # print(f"""load_retriever with top_k_vectorstore='{top_k_vectorstore}'""") # Replit console
    # Get the Retriever from the Vectorstore
    return current_vectorstore.as_retriever(
        search_kwargs={"k": top_k_vectorstore}
    )

@st.cache_resource()
def load_memory(_chat_history_resource, top_k_history): # Pass the actual chat_history resource
    # print(f"""load_memory with top-k={top_k_history}""") # Replit console
    return ConversationBufferWindowMemory(
        chat_memory=_chat_history_resource, # Use the passed resource
        return_messages=True,
        k=top_k_history,
        memory_key="chat_history",
        input_key="question",
        output_key='answer',
    )

def generate_queries_chain(_model): # Pass the model
    prompt_template = f"""You are a helpful assistant that generates multiple search queries based on a single input query in language {st.session_state.get("language", "es_ES")}.
Generate multiple search queries related to: {{original_query}}
OUTPUT (4 queries):"""
    return ChatPromptTemplate.from_messages([("system", prompt_template)]) | _model | StrOutputParser() | (lambda x: x.split("\n"))


def reciprocal_rank_fusion(results: list[list], k=60):
    from langchain.load import dumps, loads # Keep import local if only used here

    fused_scores = {}
    for docs in results:
        # Assumes the docs are returned in sorted order of relevance
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # previous_score = fused_scores[doc_str] # Unused variable
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results

# Describe the image based on OpenAI
def describeImage(image_bin, current_language): # Pass current language
    # print ("describeImage") # Replit console
    image_base64 = base64.b64encode(image_bin).decode()
    try: # Add try-except for API calls
        response = openai.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Provide a search text for the main topic of the image writen in {current_language}"},
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}",
                    },
                    },
                ],
                }
            ],
            max_tokens=100,  # Reduced for a search text
        )
        # print (f"describeImage result: {response}") # Replit console
        return response
    except Exception as e:
        st.error(f"Error describing image: {e}")
        return None


##################
### Data Cache ###
##################

# Cache localized strings
@st.cache_data()
def load_localization(locale_code):
    # print("load_localization for:", locale_code) # Replit console
    try:
        df = pd.read_csv("./customizations/localization.csv")
        df_lang = df.query(f"locale == '{locale_code}'")
        if df_lang.empty and locale_code != "en_US": # Fallback to en_US if specific lang not found
            # print(f"Locale {locale_code} not found, falling back to en_US") # Replit console
            df_lang = df.query("locale == 'en_US'")
        if df_lang.empty: # Fallback if en_US also not found (should not happen if file is correct)
             # print("Critical: en_US locale not found in localization.csv") # Replit console
             return {"error": "Default localization not found"}
        lang_dict_loaded = pd.Series(df_lang.value.values, index=df_lang.key).to_dict()
        return lang_dict_loaded
    except Exception as e:
        # print(f"Error loading localization: {e}") # Replit console
        # Provide a minimal dict for critical UI elements to prevent crashing
        return {
            "assistant_welcome": "Welcome!", "logout_caption": "Logged in as", "logout_button": "Logout",
            "assistant_question": "How can I help you?", "take_picture": "Take picture..."
        }


# Cache localized strings
@st.cache_data()
def load_rails(username_key):
    # print("load_rails for:", username_key) # Replit console
    try:
        df = pd.read_csv("./customizations/rails.csv")
        df_user = df.query(f"username == '{username_key}'")
        rails_dict_loaded = pd.Series(df_user.value.values, index=df_user.key).to_dict()
        return rails_dict_loaded
    except Exception as e:
        # print(f"Error loading rails: {e}") # Replit console
        return {} # Return empty dict if error or no rails

#############
### Login ###
#############

# Check for username/password and set the username accordingly
if not check_password():
    st.stop()  # Do not continue if check_password is not True.

# These should be set after successful login by check_password()
username = st.session_state.user
language = st.secrets.languages.get(username, "es_ES") # Default to Spanish if user has no language set
st.session_state.language = language # Store in session state for describeImage and get_prompt
lang_dict = load_localization(language)


#######################
### Resources Cache ###
#######################
if "debug_messages" not in st.session_state:
    st.session_state.debug_messages = []


# Cache OpenAI Embedding for future runs
@st.cache_resource(show_spinner=lambda: lang_dict.get('load_embedding', 'Loading Embeddings...'))
def load_embedding_cached(): # Renamed to avoid conflict with global 'embedding'
    # print("load_embedding_cached") # Replit console
    try:
        return OpenAIEmbeddings()
    except Exception as e:
        st.error(f"Failed to load OpenAI Embeddings: {e}. Check your OPENAI_API_KEY.")
        return None

# Cache Vector Store for future runs
@st.cache_resource(show_spinner=lambda: lang_dict.get('load_vectorstore', 'Loading Vector Store...'))
def load_vectorstore_cached(_embedding_resource, username_key): # Pass embedding and username
    # print(f"load_vectorstore_cached for {username_key}") # Replit console
    if not _embedding_resource:
        st.error("Embeddings not loaded, cannot initialize Vector Store.")
        return None
    try:
        # Ensure ASTRA_ENDPOINT and ASTRA_TOKEN are loaded, preferentially from st.secrets if available
        astra_endpoint = st.secrets.get("ASTRA_ENDPOINT", os.environ.get("ASTRA_ENDPOINT"))
        astra_token = st.secrets.get("ASTRA_TOKEN", os.environ.get("ASTRA_TOKEN"))

        if not all([astra_endpoint, astra_token]):
            st.error("Astra DB Endpoint or Token not configured in secrets/environment.")
            return None

        return AstraDB(
            embedding=_embedding_resource,
            collection_name=f"vector_context_{username_key}", # Use username_key
            api_endpoint=astra_endpoint,
            token=astra_token,
            # namespace=st.secrets.get("ASTRA_KEYSPACE") # Add if you use a specific keyspace from secrets
        )
    except Exception as e:
        st.error(f"Failed to initialize AstraDB Vector Store: {e}")
        return None

# Cache Chat History for future runs
@st.cache_resource(show_spinner=lambda: lang_dict.get('load_message_history', 'Loading Chat History...'))
def load_chat_history_cached(username_key): # Pass username
    # print(f"load_chat_history_cached for {username_key}_{st.session_state.session_id}") # Replit console
    try:
        # Ensure ASTRA_ENDPOINT and ASTRA_TOKEN are loaded
        astra_endpoint = st.secrets.get("ASTRA_ENDPOINT", os.environ.get("ASTRA_ENDPOINT"))
        astra_token = st.secrets.get("ASTRA_TOKEN", os.environ.get("ASTRA_TOKEN"))

        if not all([astra_endpoint, astra_token]):
            st.error("Astra DB Endpoint or Token not configured for Chat History.")
            return None # Or a dummy history object if preferred

        return AstraDBChatMessageHistory(
            session_id=f"{username_key}_{st.session_state.session_id}",
            api_endpoint=astra_endpoint,
            token=astra_token,
            # keyspace_name=st.secrets.get("ASTRA_KEYSPACE") # Add if you use a specific keyspace
        )
    except Exception as e:
        st.error(f"Failed to initialize AstraDB Chat Message History: {e}")
        return None # Or a dummy history object


#####################
### Session state ###
#####################

# Start with empty messages, stored in session state
if 'messages' not in st.session_state:
    st.session_state.messages = [AIMessage(content=lang_dict.get('assistant_welcome', "Hi! How can I help?"))]


############
### Main ###
############

# Show a custom welcome text or the default text
try:
    st.markdown(Path(f"""./customizations/welcome/{username}.md""").read_text(encoding='utf-8'))
except FileNotFoundError:
    try:
        st.markdown(Path('./customizations/welcome/default.md').read_text(encoding='utf-8'))
    except FileNotFoundError:
        st.info("Welcome message file not found.") # Fallback message
except Exception as e:
    st.warning(f"Could not load welcome message: {e}")


# Show a custom logo (svg or png) or the DataStax logo
with st.sidebar:
    logo_path_str = ""
    try:
        # Prioritize SVG then PNG for custom logo
        custom_logo_svg = Path(f"./customizations/logo/{username}.svg")
        custom_logo_png = Path(f"./customizations/logo/{username}.png")
        default_logo = Path('./customizations/logo/default.svg')

        if custom_logo_svg.is_file():
            logo_path_str = str(custom_logo_svg)
        elif custom_logo_png.is_file():
            logo_path_str = str(custom_logo_png)
        elif default_logo.is_file():
            logo_path_str = str(default_logo)
        
        if logo_path_str:
            st.image(logo_path_str, use_column_width="always")
            st.text('') # For spacing
        else:
            st.text('') # DataStax logo was here, can be removed or replaced
            st.text('')
    except Exception as e:
        st.warning(f"Could not load logo: {e}")


# Logout button
with st.sidebar:
    st.markdown(f"""{lang_dict.get('logout_caption', "Logged in as")} :orange[{username}]""")
    logout_button = st.button(lang_dict.get('logout_button', "Logout"))
    if logout_button:
        logout()

with st.sidebar:
    st.divider()

# Initialize resources after login and language is set
embedding = load_embedding_cached()
if embedding:
    vectorstore = load_vectorstore_cached(embedding, username) # Pass embedding and username
if vectorstore: # Only load chat history if vectorstore (and thus Astra creds) are OK
    chat_history = load_chat_history_cached(username) # Pass username
else:
    chat_history = None # Ensure it's None if vectorstore failed

# Options panel
with st.sidebar:
    rails_dict = load_rails(username) # Load rails based on username

    # Chat history settings
    disable_chat_history = st.toggle(lang_dict.get('disable_chat_history', "Disable Chat History?"))
    top_k_history_val = 5 if disable_chat_history else st.secrets.get("DEFAULT_TOP_K_HISTORY", {}).get(username, 5)
    top_k_history = st.slider(lang_dict.get('k_chat_history', "K for Chat History"), 1, 10, top_k_history_val, disabled=disable_chat_history)
    
    if chat_history: # Only initialize memory if chat_history is valid
        memory = load_memory(chat_history, top_k_history if not disable_chat_history else 0)
    else: # Provide a dummy memory or handle its absence if chat_history failed
        # This part needs careful handling if chat_history can be None
        # For now, let's assume if chat_history is None, memory operations might fail or need guards
        memory = None 
        st.warning("Chat history not available. Memory features might be limited.")


    if chat_history: # Only show delete button if history exists
        delete_history_button = st.button(lang_dict.get('delete_chat_history_button', "Delete Chat History"), disabled=disable_chat_history)
        if delete_history_button and memory:
            with st.spinner(lang_dict.get('deleting_chat_history', "Deleting chat history...")):
                memory.clear()
            st.success("Chat history deleted.") # Provide feedback
            st.rerun() # Rerun to clear messages from display

    # Vector store settings
    disable_vector_store = st.toggle(lang_dict.get('disable_vector_store', "Disable Vector Store?"))
    top_k_vectorstore_val = 5 if disable_vector_store else st.secrets.get("DEFAULT_TOP_K_VECTORSTORE", {}).get(username, 5)
    top_k_vectorstore = st.slider(lang_dict.get('top_k_vector_store', "Top-K for Vector Store"), 1, 10, top_k_vectorstore_val, disabled=disable_vector_store)
    
    rag_strategies = ('Basic Retrieval', 'Maximal Marginal Relevance', 'Fusion')
    default_strategy = st.secrets.get("DEFAULT_RAG_STRATEGY", {}).get(username, 'Basic Retrieval')
    strategy_index = rag_strategies.index(default_strategy) if default_strategy in rag_strategies else 0
    strategy = st.selectbox(lang_dict.get('rag_strategy', "RAG Strategy"), rag_strategies, index=strategy_index, help=lang_dict.get('rag_strategy_help', "Select RAG strategy"), disabled=disable_vector_store)


    custom_prompt_text = ''
    custom_prompt_file_path_user = Path(f"./customizations/prompt/{username}.txt")
    custom_prompt_file_path_default = Path("./customizations/prompt/default.txt")
    
    try
