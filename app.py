# === IMPORTS ===
import os
import openai
import streamlit as st
from llama_index.core import (
    SimpleDirectoryReader, VectorStoreIndex, StorageContext,
    load_index_from_storage, PromptTemplate, Settings
)
from llama_index.llms.openai import OpenAI as OpenAiLlm
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.memory import ChatSummaryMemoryBuffer
from llama_index.core.llms import ChatMessage
import tiktoken

# === INITIALISATION DE L'INTERFACE ===
# -------------------------------------

# === AFFICHAGE DU LOGO ===
st.logo(image = "media/logo.svg", size = "large", link = "https://www.eikon.ch/")

# === TITRE DE L'APPLICATION ===
st.markdown("""
    <div style='text-align: center;'>
        <h1>aikon</h1>
        <h5>Le conseiller IA d'eikon</h5>
    </div>
""", unsafe_allow_html=True)

# === INTERFACE DE CHAT ===
# créer le premier message de l'IA
if "messages" not in st.session_state:
    st.session_state.messages = [ChatMessage(role = "assistant", content = "En quoi puis-je vous aider ?")]

# ajouter le champ pour le prompt utilisateur et la logique pour son entrée
if prompt := st.chat_input("Posez votre question"):
    st.session_state.messages.append(ChatMessage(role = "user", content = prompt))

# avatars pour les bulles de chat
def set_avatar(role):
    return "media/assistant.png" if role == "assistant" else "media/user.png"

# Affichage des messages
for message in st.session_state.messages:
    with st.chat_message(message.role, avatar=set_avatar(message.role)):
        st.write(message.content)

# === UTILISATION DU LLM ===
# --------------------------

# === CONFIGURATION DE L'API OPENAI ===
openai.api_key = ""
# Pour la version de production, la clé API ne doit pas être dans le code. Il faut utiliser une variable d'environnement:
# Dans le terminal, écrire: export OPENAI_API_KEY=XXXX (remplacer les XXXX par la clé)
#                           --------------------------

# === INITIALISATION DU MODÈLE LLM ===
# modèle de chat
# disponibles pour ce projet: gpt-4.1-nano / gpt-4o-mini
model = "gpt-4o-mini"
llm = OpenAiLlm(model=model, temperature=0.01) # 0 = très précis, peu créatif // 1 = pas fiable du tout, très créatif
Settings.llm = llm
# modèle de représentation de texte
# disponibles pour ce projet: text-embedding-3-small
embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.embed_model = embed_model

# === GESTION DES DONNEES DE L'INDEX ===
# chemins des dossiers
DATA_DIR = "./source"
PROCESSED_DIR = "./processed-data"
# chargement ou création de l'index
@st.cache_data
def load_index():
    if not os.path.exists(PROCESSED_DIR):
        documents = SimpleDirectoryReader(DATA_DIR).load_data()
        index = VectorStoreIndex.from_documents(documents, embed_model=Settings.embed_model)
        index.storage_context.persist(persist_dir=PROCESSED_DIR)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=PROCESSED_DIR)
        index = load_index_from_storage(storage_context)
    return index
index = load_index()

# === RÉSUMÉ DE LA CONVERSATION ===
# initialisation du résumé
if not "summary" in st.session_state:
    st.session_state.summary = [ChatMessage(role="system", content="No summary yet"), None, None]

summarizer_llm = OpenAiLlm(model=model, max_tokens=512)
tokenizer_fn = tiktoken.get_encoding("cl100k_base").encode
memory = ChatSummaryMemoryBuffer.from_defaults(
    chat_history=st.session_state.summary,
    llm=summarizer_llm,
    token_limit=2,
    tokenizer_fn=tokenizer_fn,
)
# mise à jour du résumé
def updateSummary():
    if len(st.session_state.messages) >= 2:
        st.write(len(st.session_state.messages))
        st.session_state.summary[1] = st.session_state.messages[-2]
        st.session_state.summary[2] = st.session_state.messages[-1]
        summary = memory.get()[0]
        st.session_state.summary[0] = ChatMessage(role=summary.role, content=summary.content)

# === TEMPLATE DE PROMPT PERSONNALISÉ ===
def prepare_prompt_template(chat_summary):
    template = f"""
    Tu es aikon, le conseiller IA d'eikon. eikon est l'école professionnelle en arts appliqués à Fribourg, en Suisse.
    Tu aides les élèves, parents, enseignants et personnel administratif à comprendre les règles de l'école.
    Les mots aikon et eikon doivent s'écrire en minuscules.

    Résumé de la conversation passée :
    ----------------------
    {chat_summary}
    ----------------------

    Nouvelle question : {{query_str}}

    Identifie la langue de la question.
    Indique dans ta réponse la langue détectée.

    Voici les informations pertinentes que tu connais :
    ----------------------
    {{context_str}}
    ----------------------

    À partir de ces connaissances et du résumé, réponds à la question dans la langue que tu as indiquée.
    """
    return PromptTemplate(template)


# === TRAITEMENT DE LA NOUVELLE QUESTION ===
if st.session_state.messages[-1].role == "user":
    with st.chat_message("assistant", avatar=set_avatar("assistant")):
        with st.spinner("aikon réfléchit un moment. Merci de patienter..."):
            prompt_template = prepare_prompt_template(st.session_state.summary)
            query_engine = index.as_query_engine(text_qa_template=prompt_template)
            response = query_engine.query(prompt)

        if response:
            # Affichage de la réponse
            st.write(response.response)

            # Affichage des sources
            if hasattr(response, "source_nodes"):
                with st.expander("📚 Sources utilisées"):
                    for node in response.source_nodes:
                        st.markdown(f"- **Document**: {node.metadata.get('file_name', 'inconnu')}")
                        st.markdown(f"  > {node.node.get_text()[:200]}...")
            
            # Enregistrement de la réponse
            st.session_state.messages.append(ChatMessage(role="assistant", content=response.response))

# === MISE A JOUR DE RESUME APRES LA REPONSE ===
if st.session_state.messages[-1].role == "assistant":
    updateSummary()
