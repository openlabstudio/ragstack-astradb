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
### Globals ### # No es ideal usar globals, pero mantenemos la estructura original
###############

lang_dict = {} # Inicializar para evitar errores si no se carga
language = "es_ES" # Default
rails_dict = {} # Inicializar
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
            st.text_input('Username', key='username')
            st.text_input('Password', type='password', key='password')
            st.form_submit_button('Login', on_click=password_entered)

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        # Asegúrate de que 'passwords' exista en st.secrets
        if 'passwords' in st.secrets and \
           st.session_state.get('username') in st.secrets['passwords'] and \
           hmac.compare_digest(st.session_state.get('password', ''), st.secrets.passwords[st.session_state.get('username', '')]):
            st.session_state['password_correct'] = True
            st.session_state.user = st.session_state['username']
            if 'password' in st.session_state: # Borrar solo si existe
                del st.session_state['password']  # Don't store the password.
        else:
            st.session_state['password_correct'] = False

    if st.session_state.get('password_correct', False):
        return True

    login_form()
    if "password_correct" in st.session_state and not st.session_state.password_correct: # Mostrar error solo si el intento falló
        st.error('😕 Usuario desconocido o contraseña incorrecta')
    return False

def logout():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.cache_resource.clear()
    st.cache_data.clear()
    st.rerun()

# Function for Vectorizing uploaded data into Astra DB
def vectorize_text(uploaded_files_param, current_vectorstore, current_lang_dict): # Pasar dependencias
    for uploaded_file in uploaded_files_param:
        if uploaded_file is not None:
            temp_dir = tempfile.TemporaryDirectory()
            file = uploaded_file
            print(f"""Processing: {file.name}""")
            temp_filepath = os.path.join(temp_dir.name, file.name)
            with open(temp_filepath, 'wb') as f:
                f.write(file.getvalue())

            if uploaded_file.name.endswith('txt'):
                with open(temp_filepath, 'r', encoding='utf-8') as f_txt:
                    file_content_list = [f_txt.read()]
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
                texts = text_splitter.create_documents(file_content_list, [{'source': uploaded_file.name}] * len(file_content_list))
                if current_vectorstore: current_vectorstore.add_documents(texts)
                st.info(f"{len(texts)} {current_lang_dict.get('load_text', 'text segments loaded')}")
            
            elif uploaded_file.name.endswith('pdf'):
                docs = []
                loader = PyPDFLoader(temp_filepath)
                docs.extend(loader.load())
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
                pages = text_splitter.split_documents(docs)
                if current_vectorstore: current_vectorstore.add_documents(pages)  
                st.info(f"{len(pages)} {current_lang_dict.get('load_pdf', 'PDF pages/segments loaded')}")

            elif uploaded_file.name.endswith('csv'):
                docs = []
                loader = CSVLoader(temp_filepath, encoding='utf-8')
                docs.extend(loader.load())
                if current_vectorstore: current_vectorstore.add_documents(docs)
                st.info(f"{len(docs)} {current_lang_dict.get('load_csv', 'CSV documents/rows loaded')}")
            
            temp_dir.cleanup()

# Load data from URLs
def vectorize_url(urls_list_param, current_vectorstore, current_lang_dict): # Pasar dependencias
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    for url_item in urls_list_param:
        url = url_item.strip()
        if not url: continue
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()    
            pages = text_splitter.split_documents(docs)
            print (f"Loading from URL: {pages}")
            if current_vectorstore: current_vectorstore.add_documents(pages)  
            st.info(f"{len(pages)} {current_lang_dict.get('url_pages_loaded', 'pages loaded')}")
        except Exception as e:
            st.error(f"{current_lang_dict.get('url_error', 'An error occurred loading URL')} {url}: {e}")

# Define the prompt
def get_prompt(prompt_type_param, current_custom_prompt, current_language_code): # Pasar dependencias
    template = ''
    if prompt_type_param == 'Extended results':
        print ("Prompt type: Extended results")
        template = f"""You're a helpful AI assistant tasked to answer the user's questions.
# ... (resto de tu prompt extenso, asegurándote que {current_language_code} esté bien)
Use the following context to answer the question:
{{context}}
Use the following chat history to answer the question:
{{chat_history}}
Question:
{{question}}
Answer in {current_language_code}:"""

    elif prompt_type_param == 'Short results':
        print ("Prompt type: Short results")
        template = f"""You're a helpful AI assistant tasked to answer the user's questions.
# ... (resto de tu prompt corto, asegurándote que {current_language_code} esté bien)
Use the following context to answer the question:
{{context}}
Use the following chat history to answer the question:
{{chat_history}}
Question:
{{question}}
Answer in {current_language_code}:"""

    elif prompt_type_param == 'Custom':
        print ("Prompt type: Custom")
        template = current_custom_prompt
    return ChatPromptTemplate.from_messages([("system", template)])

# Get the OpenAI Chat Model
@st.cache_resource()
def load_model_cached():
    print(f"""load_model""")
    return ChatOpenAI(temperature=0.3, model='gpt-4o', streaming=True, verbose=False)

# Get the Retriever
def load_retriever_fn(current_vectorstore, top_k_vs_param):
    if not current_vectorstore: return None # Añadido chequeo
    print(f"""load_retriever with top_k_vectorstore='{top_k_vs_param}'""")
    return current_vectorstore.as_retriever(search_kwargs={"k": top_k_vs_param})

@st.cache_resource()
def load_memory_cached(_chat_history_resource, top_k_hist_param):
    if not _chat_history_resource: return None # Añadido chequeo
    print(f"""load_memory with top-k={top_k_hist_param}""")
    return ConversationBufferWindowMemory(
        chat_memory=_chat_history_resource, return_messages=True, k=top_k_hist_param,
        memory_key="chat_history", input_key="question", output_key='answer')

def generate_queries_chain_fn(_model, current_language_code): # _model es la instancia del LLM
    if not _model: return None # Añadido chequeo
    prompt_str = f"""You are a helpful assistant that generates multiple search queries based on a single input query in language {current_language_code}.
Generate multiple search queries related to: {{original_query}}
OUTPUT (4 queries):"""
    return ChatPromptTemplate.from_messages([("system", prompt_str)]) | _model | StrOutputParser() | (lambda x: x.split("\n"))

def reciprocal_rank_fusion(results: list[list], k_param=60):
    from langchain.load import dumps, loads
    fused_scores = {}
    for docs_list_item in results: # Renombrado para claridad
        if not isinstance(docs_list_item, list): continue # Asegurar que es una lista de documentos
        for rank, doc in enumerate(docs_list_item):
            try:
                doc_str = dumps(doc)
                if doc_str not in fused_scores: fused_scores[doc_str] = 0
                fused_scores[doc_str] += 1 / (rank + k_param)
            except Exception: # Capturar error si doc no es serializable
                print(f"Warning: Could not serialize document for RRF: {doc}")
                continue
    reranked_results = [(loads(doc_s), score) for doc_s, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]
    return reranked_results

def describeImage(image_bin_param, lang_param):
    print ("describeImage")
    image_base64 = base64.b64encode(image_bin_param).decode()
    try:
        response = openai.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": f"Provide a search text for the main topic of the image writen in {lang_param}"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ]}], max_tokens=100)
        print (f"describeImage result: {response}")
        return response
    except Exception as e:
        st.error(f"Error describing image: {e}")
        return None

##################
### Data Cache ###
##################
@st.cache_data()
def load_localization(locale_param):
    print(f"load_localization for: {locale_param}")
    try:
        df = pd.read_csv("./customizations/localization.csv")
        df_lang = df.query(f"locale == '{locale_param}'")
        if df_lang.empty and locale_param != "en_US":
            print(f"Locale {locale_param} not found, falling back to en_US")
            df_lang = df.query("locale == 'en_US'")
        if df_lang.empty:
             print("Critical: en_US locale not found in localization.csv. Using hardcoded defaults.")
             return {"assistant_welcome": "Welcome!", "logout_caption": "User", "logout_button": "Logout", "assistant_question":"Question..."} # Minimal defaults
        lang_dict_res = pd.Series(df_lang.value.values, index=df_lang.key).to_dict()
        return lang_dict_res
    except Exception as e:
        print(f"Error loading localization.csv: {e}. Using hardcoded defaults.")
        return {"assistant_welcome": "Welcome!", "logout_caption": "User", "logout_button": "Logout", "assistant_question":"Question..."}

@st.cache_data()
def load_rails(username_param):
    print(f"load_rails for: {username_param}")
    try:
        df = pd.read_csv("./customizations/rails.csv")
        df = df.query(f"username == '{username_param}'")
        rails_dict_res = pd.Series(df.value.values, index=df.key).to_dict()
        return rails_dict_res
    except FileNotFoundError:
        print("Warning: rails.csv not found. No rails will be loaded.")
        return {}
    except Exception as e:
        print(f"Error loading rails for {username_param}: {e}")
        return {}


#############
### Login ###
#############
if not check_password():
    st.stop()

username = st.session_state.user
language = st.secrets.get("languages", {}).get(username, "es_ES") # Default a es_ES
lang_dict = load_localization(language)

#######################
### Resources Cache ###
#######################
embedding = load_embedding_cached()
vectorstore = load_vectorstore_cached(embedding, username) if embedding else None
chat_history = load_chat_history_cached(username) if vectorstore else None # chat history depends on Astra creds often via vectorstore init
memory = load_memory_cached(chat_history, 5) if chat_history else None # Default K for memory, adjust as needed

#####################
### Session state ###
#####################
if 'messages' not in st.session_state:
    st.session_state.messages = [AIMessage(content=lang_dict.get('assistant_welcome', "Welcome!"))]

############
### Main ###
############
try:
    welcome_file_path = Path(f"./customizations/welcome/{username}.md")
    default_welcome_file_path = Path('./customizations/welcome/default.md')
    if welcome_file_path.is_file():
        st.markdown(welcome_file_path.read_text(encoding='utf-8'))
    elif default_welcome_file_path.is_file():
        st.markdown(default_welcome_file_path.read_text(encoding='utf-8'))
    else:
        st.markdown(lang_dict.get('assistant_welcome', "Welcome!")) # Fallback if no md files
except Exception as e:
    st.warning(f"Could not load welcome message: {e}")
    st.markdown(lang_dict.get('assistant_welcome', "Welcome!"))


with st.sidebar:
    # Logo display logic (simplified)
    user_logo_path_svg = Path(f"./customizations/logo/{username}.svg")
    user_logo_path_png = Path(f"./customizations/logo/{username}.png")
    default_logo_path = Path('./customizations/logo/default.svg')
    
    logo_to_display = None
    if user_logo_path_svg.is_file():
        logo_to_display = str(user_logo_path_svg)
    elif user_logo_path_png.is_file():
        logo_to_display = str(user_logo_path_png)
    elif default_logo_path.is_file():
        logo_to_display = str(default_logo_path)

    if logo_to_display:
        st.image(logo_to_display, use_column_width="always")
    else:
        st.text("Logo") # Placeholder if no logo found
    st.text('')


with st.sidebar:
    st.markdown(f"""{lang_dict.get('logout_caption', "Logged in as")} :orange[{username}]""")
    logout_button = st.button(lang_dict.get('logout_button', "Logout"))
    if logout_button:
        logout()

with st.sidebar:
    st.divider()

rails_dict = load_rails(username) # rails_dict loaded here

with st.sidebar:
    disable_chat_history = st.toggle(lang_dict.get('disable_chat_history', "Disable Chat History"), key="ch_toggle")
    top_k_history = st.slider(lang_dict.get('k_chat_history', "K for Chat History"), 1, 10, 5, disabled=disable_chat_history, key="ch_slider")
    
    if chat_history and memory: # Ensure memory is not None
        delete_history = st.button(lang_dict.get('delete_chat_history_button', "Delete Chat History"), disabled=disable_chat_history, key="del_ch_button")
        if delete_history:
            with st.spinner(lang_dict.get('deleting_chat_history', "Deleting...")):
                memory.clear()
            st.rerun()

    disable_vector_store = st.toggle(lang_dict.get('disable_vector_store', "Disable Vector Store?"), key="vs_toggle")
    top_k_vectorstore = st.slider(lang_dict.get('top_k_vector_store', "Top-K for Vector Store"), 1, 10, 5, disabled=disable_vector_store, key="vs_slider")
    
    rag_strategy_options = ('Basic Retrieval', 'Maximal Marginal Relevance', 'Fusion') # Define options
    strategy = st.selectbox(lang_dict.get('rag_strategy', "RAG Strategy"), rag_strategy_options, help=lang_dict.get('rag_strategy_help',"Help"), disabled=disable_vector_store, key="rag_select")

    custom_prompt_text_val = "" # Initialize
    custom_prompt_idx_val = 0 # Default to first option
    try:
        user_prompt_file = Path(f"./customizations/prompt/{username}.txt")
        default_prompt_file = Path("./customizations/prompt/default.txt")
        if user_prompt_file.is_file():
            custom_prompt_text_val = user_prompt_file.read_text(encoding='utf-8')
            custom_prompt_idx_val = 2 # 'Custom'
        elif default_prompt_file.is_file():
            custom_prompt_text_val = default_prompt_file.read_text(encoding='utf-8')
            # Determine index based on what default.txt implies, or set to 0 for 'Short results'
        else:
            # Provide a fallback if no prompt files are found
            custom_prompt_text_val = "You are a helpful AI assistant. Answer based on context.\nContext: {context}\nHistory: {chat_history}\nQuestion: {question}"
    except Exception as e:
        print(f"Error loading prompt text: {e}")
        custom_prompt_text_val = "Error loading prompt text."


    prompt_type_options = ('Short results', 'Extended results', 'Custom')
    prompt_type = st.selectbox(lang_dict.get('system_prompt', "System Prompt"), prompt_type_options, index=custom_prompt_idx_val, key="prompt_type_select")
    custom_prompt = st.text_area(lang_dict.get('custom_prompt', "Custom Prompt"), custom_prompt_text_val, help=lang_dict.get('custom_prompt_help', "Help"), disabled=(prompt_type != 'Custom'), key="custom_prompt_area")
    print(f"""Sidebar state: DS_VS={disable_vector_store}, K_Hist={top_k_history}, K_VS={top_k_vectorstore}, Strat={strategy}, PromptT={prompt_type}""")

with st.sidebar:
    st.divider()

# --- BLOQUES DE INGESTA COMENTADOS ---
# # Include the upload form for new data to be Vectorized
# with st.sidebar:
#     uploaded_files = st.file_uploader(lang_dict.get('load_context', "Upload context"), type=['txt', 'pdf', 'csv'], accept_multiple_files=True)
#     upload = st.button(lang_dict.get('load_context_button', "Load context button"))
#     if upload and uploaded_files:
#         if vectorstore:
#             vectorize_text(uploaded_files, vectorstore, lang_dict) # Pass dependencies
#         else:
#             st.error("Vectorstore not initialized. Cannot upload files.")


# # Include the upload form for URLs be Vectorized
# with st.sidebar:
#     urls_raw_input = st.text_area(lang_dict.get('load_from_urls', "Load from URLs"), help=lang_dict.get('load_from_urls_help', "URLs help"))
#     # Check if urls_raw_input is not None and not empty before splitting
#     urls_list_from_input = [url.strip() for url in urls_raw_input.split(',')] if urls_raw_input and urls_raw_input.strip() else []
#     upload_urls = st.button(lang_dict.get('load_from_urls_button', "Load from URLs button"))
#     if upload_urls and urls_list_from_input:
#         if vectorstore:
#             vectorize_url(urls_list_from_input, vectorstore, lang_dict) # Pass dependencies
#         else:
#             st.error("Vectorstore not initialized. Cannot upload URLs.")

# # Drop the vector data and start from scratch
# # This is controlled by secrets.toml: [delete_option] username = "True" or "False"
# delete_option_is_true = False
# if 'delete_option' in st.secrets and username in st.secrets['delete_option']:
#     delete_option_is_true = str(st.secrets.delete_option[username]).lower() == 'true'

# if delete_option_is_true:
#     with st.sidebar:
#         st.caption(lang_dict.get('delete_context', "Delete Context Caption"))
#         submitted_delete_button = st.button(lang_dict.get('delete_context_button', "Delete Context Button")) # Renamed var
#         if submitted_delete_button:
#             if vectorstore and memory: # Check both exist
#                 with st.spinner(lang_dict.get('deleting_context', "Deleting...")):
#                     vectorstore.clear()
#                     memory.clear()
#                     st.session_state.messages = [AIMessage(content=lang_dict.get('assistant_welcome', "Welcome!"))]
#                 st.rerun() # Rerun to reflect changes
#             else:
#                 st.error("Cannot delete context: Vectorstore or Memory not available.")
# --- FIN DE BLOQUES COMENTADOS ---


with st.sidebar:
    st.divider()

# Draw rails
with st.sidebar:
    st.subheader(lang_dict.get('rails_1', "Suggestions"))
    st.caption(lang_dict.get('rails_2', "Try asking:"))
    if rails_dict:
        for i in sorted(rails_dict.keys()):
            st.markdown(f"{i}. {rails_dict[i]}")
    else:
        st.markdown(lang_dict.get('no_rails_available', "No suggestions available."))


# Draw all messages, both user and agent so far
for message in st.session_state.messages:
    if hasattr(message, 'type') and hasattr(message, 'content'):
        st.chat_message(message.type).markdown(message.content)


# Now get a prompt from a user
question = st.chat_input(lang_dict.get('assistant_question', "Ask something..."))

with st.sidebar:
    st.divider()
    picture = st.camera_input(lang_dict.get('take_picture', "Take a picture"))
    if picture:
        if 'OPENAI_API_KEY' in st.secrets and st.secrets['OPENAI_API_KEY']: # Check key exists
            img_desc_response = describeImage(picture.getvalue(), language) # Pass language
            if img_desc_response and img_desc_response.choices and img_desc_response.choices[0].message.content:
                picture_desc_text = img_desc_response.choices[0].message.content # Renamed
                if not question: # If no text question, use image description
                    question = picture_desc_text
                    st.info(f"Using image description as question: {question[:100]}...") # Show snippet
            else:
                st.error("Could not get description from image.")
        else:
            st.error("OpenAI API Key not configured. Camera feature disabled.")


if question:
    print(f"Got question: {question}")
    st.session_state.messages.append(HumanMessage(content=question))
    with st.chat_message('human'):
        st.markdown(question)

    model = load_model_cached() # Use cached model
    retriever = None
    relevant_documents = []
    
    if vectorstore and not disable_vector_store:
        retriever = load_retriever_fn(vectorstore, top_k_vectorstore)
    
    if retriever and not disable_vector_store:
        if strategy == 'Basic Retrieval':
            relevant_documents = retriever.get_relevant_documents(query=question)
        elif strategy == 'Maximal Marginal Relevance':
            relevant_documents = vectorstore.max_marginal_relevance_search(query=question, k=top_k_vectorstore)
        elif strategy == 'Fusion':
            if model: # Model needed for query generation
                queries_chain_instance = generate_queries_chain_fn(model, language) # Pass model, language
                if queries_chain_instance:
                    fusion_queries_list = queries_chain_instance.invoke({"original_query": question})
                    print(f"""Fusion queries: {fusion_queries_list}""")
                    
                    fusion_queries_display = f"""*{lang_dict.get('using_fusion_queries', "Using fusion queries:")}*\n"""
                    for i, fq_item_text in enumerate(fusion_queries_list): # Iterate over text items
                        fusion_queries_display += f"""\n{i+1}. :orange[{fq_item_text}]"""
                    
                    with st.chat_message('assistant'): # Intermediate message for fusion queries
                        st.markdown(fusion_queries_display)
                    st.session_state.messages.append(AIMessage(content=fusion_queries_display))

                    retrieved_docs_lists = retriever.map().invoke(fusion_queries_list) # list of lists of docs
                    fused_docs_with_scores = reciprocal_rank_fusion(retrieved_docs_lists)
                    relevant_documents = [doc_tuple[0] for doc_tuple in fused_docs_with_scores][:top_k_vectorstore]
                    print(f"""Fusion results: {relevant_documents}""")
                else: # Fallback if query generation chain failed
                    st.warning("Fusion query generation failed. Using Basic Retrieval.")
                    relevant_documents = retriever.get_relevant_documents(query=question)
            else: # Fallback if model not available
                st.warning("Model not available for Fusion. Using Basic Retrieval.")
                relevant_documents = retriever.get_relevant_documents(query=question)

    with st.chat_message('assistant'):
        final_response_str = '' # Renamed
        response_placeholder = st.empty()
        history_for_llm = {"chat_history": []} # Renamed
        if memory:
            history_for_llm = memory.load_memory_variables({})
        print(f"Using memory: {history_for_llm}")

        if model:
            inputs_map = RunnableMap({
                'context': lambda x: x['context'],
                'chat_history': lambda x: x['chat_history'],
                'question': lambda x: x['question']
            })
            chain = inputs_map | get_prompt(prompt_type, custom_prompt, language) | model # Pass custom_prompt and language

            try:
                llm_response_obj = chain.invoke( # Renamed
                    {'question': question, 'chat_history': history_for_llm.get('chat_history',[]), 'context': relevant_documents}, 
                    config={'callbacks': [StreamHandler(response_placeholder)]}
                )
                final_response_str = llm_response_obj.content if hasattr(llm_response_obj, 'content') else str(llm_response_obj)
            except Exception as e:
                print(f"Error invoking LLM chain: {e}")
                st.error(f"Error during LLM chain invocation: {e}")
                final_response_str = lang_dict.get("error_llm_invocation", "Sorry, error during response generation.")
        else:
            final_response_str = lang_dict.get("model_unavailable_error", "AI Model not available.")
            st.warning(final_response_str)


        if memory:
            memory.save_context({'question': question}, {'answer': final_response_str})

        full_display_message = final_response_str # Renamed
        if not disable_vector_store and relevant_documents:
            full_display_message += f"""\n\n*{lang_dict.get('sources_used', "Sources used:")}*"""
            displayed_sources = [] # Renamed
            for doc in relevant_documents:
                source_name_meta = doc.metadata.get('source', 'Unknown Source') if hasattr(doc, 'metadata') and doc.metadata else 'Unknown Source'
                source_basename_display = os.path.basename(os.path.normpath(source_name_meta)) # Renamed
                if source_basename_display not in displayed_sources:
                    full_display_message += f"""\n📙 :orange[{source_basename_display}]"""
                    displayed_sources.append(source_basename_display)
        elif not disable_vector_store: # but relevant_documents is empty
             full_display_message += f"""\n\n*{lang_dict.get('no_relevant_docs_found', "No specific documents found for this query.")}*"""


        if not disable_chat_history and memory and history_for_llm.get('chat_history',[]):
            num_hist_pairs = len(history_for_llm['chat_history']) // 2
            full_display_message += f"""\n\n*{lang_dict.get('chat_history_info', "Chat history considered")}: ({num_hist_pairs} / {top_k_history})*"""
        
        response_placeholder.markdown(full_display_message)
        st.session_state.messages.append(AIMessage(content=full_display_message))

with st.sidebar:
    st.divider()
    st.caption(lang_dict.get("app_version", "v0.1.0_mod"))
