# Importation des modules nécessaires
# Ces modules sont des "boîtes à outils" qui permettent d'ajouter des fonctionnalités à Python
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage  # Pour gérer les documents et l'indexation
from llama_index.core import PromptTemplate, Settings  # Pour créer des modèles de questions/réponses et configurer le système
from llama_index.llms.openai import OpenAI  # Pour utiliser l'intelligence artificielle d'OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding  # Pour utiliser les embeddings d'OpenAI
import os  # Pour interagir avec le système de fichiers (dossiers, fichiers, etc.)
import streamlit as st  # Pour créer une interface web simple

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

# Chemin du dossier où sont stockés les documents à analyser
DATA_DIR = "./data"
# Chemin du dossier où sera sauvegardé l'index (la "mémoire" de l'IA)
INDEX_DIR = "./storage"

# Vérification de la clé API OpenAI
if not os.getenv("OPENAI_API_KEY"):
    st.error("⚠️ Clé API OpenAI manquante ! Veuillez définir la variable d'environnement OPENAI_API_KEY")
    st.stop()

# === INITIALISATION DU MODÈLE LLM ===
# Nom du modèle d'intelligence artificielle à utiliser (ici, un modèle OpenAI)
# disponibles pour ce projet: gpt-4.1-nano / gpt-4o-mini
LLM_MODEL_NAME = "gpt-4o-mini"


# On crée une instance du modèle d'OpenAI avec le nom choisi
llm = OpenAI(model = LLM_MODEL_NAME, temperature=0.01) # 0 = très précis, peu créatif // 1 = pas fiable du tout, très créatif
# On indique à llama-index d'utiliser ce modèle par défaut
Settings.llm = llm

# modèle de représentation de texte
# disponibles pour ce projet: text-embedding-3-small
embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.embed_model = embed_model

# Cette fonction permet de charger ou de créer l'index des documents
@st.cache_data  # Cette ligne permet de ne pas refaire le travail si rien n'a changé (gain de temps)
def load_index():
    """
    Charge ou crée un index à partir des documents du répertoire spécifié.

    Si le dossier d'index n'existe pas, le système lit les documents du dossier data,
    crée un nouvel index et le sauvegarde. Si le dossier d'index existe, il charge
    l'index depuis le stockage directement.
    """
    # Si le dossier d'index n'existe pas, on lit les documents et on crée l'index
    if not os.path.exists(INDEX_DIR):
        documents = SimpleDirectoryReader(DATA_DIR).load_data()  # On lit tous les fichiers du dossier data
        index = VectorStoreIndex.from_documents(documents)  # On crée l'index à partir des documents
        index.storage_context.persist(persist_dir=INDEX_DIR)  # On sauvegarde l'index dans le dossier storage
    else:
        # Si l'index existe déjà, on le charge pour ne pas tout refaire
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
        index = load_index_from_storage(storage_context)
    return index  # On retourne l'index (la "mémoire" de l'IA)

# On charge l'index au démarrage de l'application
index = load_index()

# Cette fonction prépare le modèle de question/réponse (le "prompt")
def prepare_template():
    """
    Prépare un template de prompt pour le système de questions/réponses.
    """
    # Le texte ci-dessous sert de consigne à l'IA pour répondre comme on le souhaite
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
    # Ne réponds pas si la question porte sur un sujet qui est en dehors des connaissances de l'école.
    # On pourrait ajouter une blague à la fin de la réponse (optionnel)
    # if random.random() < 0.5:
    #     text_qa_template_str += "Termine par une blague geek."
    qa_template = PromptTemplate(text_qa_template_str) # On crée le template
    return qa_template  # On retourne le template

# On affiche un titre et une description sur la page web
st.markdown("""
# AI-kon

AI-kon est un assistant virtuel qui répond aux questions des élèves d'eikon. Il est basé sur le modèle GPT-4o-mini de OpenAI et utilise la bibliothèque llama-index pour gérer les documents et les requêtes. Il connait tout sur l'école, son règlement et son fonctionnement.
"""
)

# On vérifie si la liste des messages existe déjà dans la session (pour garder l'historique du chat)
if "messages" not in st.session_state:
    # Si ce n'est pas le cas, on l'initialise avec un message d'accueil de l'assistant
    st.session_state.messages = [{"role": "assistant", "content": "Pose tes questions sur l'école, je suis là pour t'aider !"}]

# On récupère la question de l'utilisateur (si elle existe) et on l'ajoute à la liste des messages
if prompt := st.chat_input("Que veux-tu savoir ?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

# On définit les avatars (icônes) pour l'assistant et l'utilisateur
assistant_avatar_filepath = "🤖"
user_avatar_filepath = "🙂"
# On affiche tous les messages du chat avec l'avatar correspondant
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=assistant_avatar_filepath if message["role"] == "assistant" else user_avatar_filepath):
        st.write(message["content"])

# On prépare le template de question/réponse
qa_template = prepare_template()
# On crée un "moteur de recherche" qui va utiliser l'index et le template pour répondre
query_engine = index.as_query_engine(text_qa_template=qa_template, similarity_top_k=2)

# Si le dernier message vient de l'utilisateur, l'assistant doit répondre
if st.session_state.messages[-1]["role"] == "user":
    # On affiche le message de l'assistant avec l'avatar
    with st.chat_message("assistant", avatar=assistant_avatar_filepath):
        # On affiche un message d'attente pendant que l'IA réfléchit
        with st.spinner("Patientez deux secondes le temps que AI-kon se réveille"):
            response = query_engine.query(prompt)  # L'IA génère une réponse à la question
        if response:
            st.markdown(response.response)  # On affiche la réponse sur la page web
            # On ajoute la réponse à l'historique des messages
            st.session_state.messages.append({"role": "assistant", "content": response.response})

        # Pour voir le texte utilisé par l'IA pour répondre (optionnel, pour les curieux) :
        if hasattr(response, "source_nodes"):
            with st.expander("📚 Sources utilisées"):
                for node in response.source_nodes:
                    st.markdown(f"- **Document**: {node.metadata.get('file_name', 'inconnu')}")
                    st.markdown(f"  > {node.node.get_text()[:200]}...")

            

