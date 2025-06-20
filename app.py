from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage  
from llama_index.core import PromptTemplate, Settings  
from llama_index.llms.openai import OpenAI  
from llama_index.embeddings.openai import OpenAIEmbedding  
import os  
import streamlit as st  

# ---------------------------------------
# CONFIGURATION et INSTALLATION
# ---------------------------------------

# Avant d'utiliser ce script, il faut installer les bibliothèques nécessaires :
# - Pour installer llama-index, tapez dans le terminal : pip install llama-index
# - Pour installer streamlit, tapez dans le terminal : pip install streamlit
#
# Pour utiliser l'intelligence artificielle d'OpenAI, il faut une clé API (une sorte de mot de passe) :
# - Créez un compte sur https://platform.openai.com/api-keys et copiez votre clé
# - Sur Mac ou Linux, tapez dans le terminal : export OPENAI_API_KEY=VOTRE_CLÉ
# - Sur Windows, tapez : set OPENAI_API_KEY=VOTRE_CLÉ
#
# Pour lancer l'application, tapez dans le terminal : streamlit run app.py

DATA_DIR = "./data"
INDEX_DIR = "./storage"

if not os.getenv("OPENAI_API_KEY"):
    st.error("⚠️ Clé API OpenAI manquante ! Veuillez définir la variable d'environnement OPENAI_API_KEY")
    st.stop()

# disponibles pour ce projet: gpt-4.1-nano / gpt-4o-mini
LLM_MODEL_NAME = "gpt-4o-mini"


llm = OpenAI(model = LLM_MODEL_NAME, temperature=0.01) 
Settings.llm = llm

# disponibles pour ce projet: text-embedding-3-small
embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.embed_model = embed_model

@st.cache_data  
def load_index():
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
    text_qa_template_str = """
    Tu es AI-kon, un expert en interactive media design et tu es à eikon, une école professionnelle d'arts appliqués à Fribourg, en Suisse. Tu connais tout sur l'école, son règlement et son fonctionnement, ainsi que sur les métiers de la création numérique.
    Tu réponds aux questions des élèves, en les tutoyant.
    Un·e élève t'a posé cette question : {query_str}
    Voilà tout ce que tu sais à ce sujet :
    --------
    {context_str}
    --------
    Détecte la langue de la question posée par l'élève et répond dans cette langue. Voici les règles à suivre pour répondre :
    À partir de ces connaissances, et uniquement à partir d'elles. Ne réponds pas à la question si tu n'as pas d'informations pertinentes. Si tu ne sais pas, dis que tu ne sais pas.
    Réponds de manière concise et précise, sans faire de blabla inutile. Sois amical·e et engageant·e, mais reste professionnel·le. Utilise un langage simple et clair, sans jargon technique.
    """
    qa_template = PromptTemplate(text_qa_template_str) 
    return qa_template  

st.markdown("""
# AI-kon

AI-kon est un assistant virtuel qui répond aux questions des élèves d'eikon. Il est basé sur le modèle GPT-4o-mini de OpenAI et utilise la bibliothèque llama-index pour gérer les documents et les requêtes. Il connait tout sur l'école, son règlement et son fonctionnement.
"""
)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Pose tes questions sur l'école, je suis là pour t'aider !"}]

if prompt := st.chat_input("Que veux-tu savoir ?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

assistant_avatar_filepath = "🤖"
user_avatar_filepath = "🙂"
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

        if hasattr(response, "source_nodes"):
            with st.expander("📚 Sources utilisées"):
                for node in response.source_nodes:
                    st.markdown(f"- **Document**: {node.metadata.get('file_name', 'inconnu')}")
                    st.markdown(f"  > {node.node.get_text()[:200]}...")

            

