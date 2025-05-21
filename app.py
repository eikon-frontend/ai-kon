from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core import PromptTemplate, Settings
from llama_index.llms.openai import OpenAI
import os
import streamlit as st
import random

# ---------------------------------------
# CONFIGURATION et INSTALLATION
# ---------------------------------------

# Installer llama-index : pip install llama-index
# Installer streamlit : pip install streamlit
# Placez la clé OpenAI dans une variable d'environnement nommée OPENAI_API_KEY :
# - "export OPENAI_API_KEY=XXXXX" sur linux et macOS, "set OPENAI_API_KEY=XXXXX" sur Windows
# lancez le script avec : streamlit run app.py

DATA_DIR = "./data"
INDEX_DIR = "./storage"
LLM_MODEL_NAME = "gpt-4o-mini"

llm = OpenAI(model = LLM_MODEL_NAME)
Settings.llm = llm

@st.cache_data
def load_index():
    """
    Charge ou crée un index à partir des documents du répertoire spécifié.

    Si le dossier d'index n'existe pas, le système lit les documents du dossier data,
    crée un nouvel index et le sauvegarde. Si le dossier d'index existe, il charge
    l'index depuis le stockage directement.
    """
    if not os.path.exists(INDEX_DIR):
        documents = SimpleDirectoryReader(DATA_DIR).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=INDEX_DIR)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
        index = load_index_from_storage(storage_context)
    return index

index = load_index()

def prepare_template():
    """
    Prépare un template de prompt pour le système de questions/réponses.
    """
    text_qa_template_str = """
    Tu es AI-kon, un expert en interactive media design et tu es à eikon, une école professionnelle d'arts appliqués à Fribourg, en Suisse. Tu connais tout sur l'école, son règlement et son fonctionnement, ainsi que sur les métiers de la création numérique.
    Tu réponds aux questions des élèves, en les tutoyant.
    Un·e élève t'a posé cette question : {query_str}
    Voilà tout ce que tu sais à ce sujet :
    --------
    {context_str}
    --------
    À partir de ces connaissances, et uniquement à partir d’elles, réponds en français à la question.
    Réponds en faisant des allégories et des métaphores alambiquées, comme si tu étais un·e poète·esse du 19ème siècle.
    """
    # Blague finale :
    # if random.random() < 0.5:
    #     text_qa_template_str += "Termine par une blague geek."
    qa_template = PromptTemplate(text_qa_template_str)
    return qa_template


st.markdown("""
# AI-kon

AI-kon est un assistant virtuel qui répond aux questions des élèves d'eikon. Il est basé sur le modèle GPT-4o-mini de OpenAI et utilise la bibliothèque llama-index pour gérer les documents et les requêtes. Il connait tout sur l'école, son règlement et son fonctionnement.
"""
)

# Initialise les messages de session_state s'ils ne sont pas déjà présents
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Pose tes questions sur l'école, je suis là pour t'aider !"}]

# Capture l'entrée utilisateur et l'ajoute aux messages de session_state
if prompt := st.chat_input("Que veux-tu savoir ?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

# assistant_avatar_filepath = "media/avatar.png"
assistant_avatar_filepath = "🤖"
user_avatar_filepath = "🙂"
# Affiche les messages du chat avec les avatars appropriés
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=assistant_avatar_filepath if message["role"] == "assistant" else user_avatar_filepath):
        st.write(message["content"])


qa_template = prepare_template()
query_engine = index.as_query_engine(text_qa_template=qa_template, similarity_top_k=2)

if st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant", avatar=assistant_avatar_filepath):
        with st.spinner("Patientez deux secondes le temps que AI-kon se réveille"):
            response = query_engine.query(prompt)
        if response:
            st.markdown(response.response)
            st.session_state.messages.append({"role": "assistant", "content": response.response})

            # pour afficher le contenu utilisé pour générer la réponse :
            #for node in response.source_nodes:
            #    print("\n----------------")
            #    print(f"Texte utilisé pour répondre : {node.text}")

