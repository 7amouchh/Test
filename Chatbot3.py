import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings  # Utilisation de SentenceTransformer
import subprocess
import tempfile

# Définir le chemin vers l'exécutable Ollama et le modèle LLaMA
OLLAMA_PATH = "C:/Users/mayadi/AppData/Local/Programs/Ollama/ollama.exe"
LLAMA_MODEL = "llama3.2"

# Créer l'objet d'embedding avec le modèle SentenceTransformer
embedding_model = SentenceTransformerEmbeddings(model_name='paraphrase-MiniLM-L6-v2')

# Télécharger des fichiers PDF
st.header("Mon premier Chatbot")

with st.sidebar:
    st.title("Vos Documents")
    files = st.file_uploader("Téléchargez des fichiers PDF et commencez à poser des questions", type="pdf", accept_multiple_files=True)

# Extraire le texte
if files is not None:
    all_texts = []
    
    for file in files:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:  # Vérifiez si le texte n'est pas vide
                text += page_text
        if text:  # Ajouter le texte du fichier si non vide
            all_texts.append(text)

    # Combiner le texte de tous les fichiers PDF
    combined_text = "\n".join(all_texts)

    # Diviser le texte en morceaux
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(combined_text)

    # Vérification si les chunks ne sont pas vides
    if not chunks:
        st.write("Aucun texte n'a été extrait des fichiers PDF. Veuillez vérifier le contenu des fichiers.")
    else:
        # Créer la boutique vectorielle - FAISS
        vector_store = FAISS.from_texts(chunks, embedding_model)

        # Obtenir la question de l'utilisateur
        user_question = st.text_input("Tapez votre question ici")

        # Effectuer une recherche de similarité et obtenir des réponses
        if user_question:
            # Rechercher les morceaux les plus similaires
            match = vector_store.similarity_search(user_question)

            # Préparer le contexte pour LLaMA
            context = " ".join([doc.page_content for doc in match])  # Joindre les textes similaires

            # Appeler le modèle LLaMA à l'aide de subprocess
            with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') as tmp_out, \
                 tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') as tmp_err:

                # Construire l'entrée pour le modèle LLaMA
                input_text = f"Voici le contexte :\n{context}\n\nQuestion : {user_question}\nRéponse :"

                # Envoyer l'entrée à Ollama via stdin
                process = subprocess.Popen(
                    [OLLAMA_PATH, "run", LLAMA_MODEL],
                    stdin=subprocess.PIPE,  # Utiliser stdin pour envoyer la question
                    stdout=tmp_out,
                    stderr=tmp_err,
                    text=True
                )

                # Écrire l'entrée dans stdin
                process.communicate(input_text)

                # Lire les sorties
                output = ""
                error = ""
                with open(tmp_out.name, 'r', encoding='utf-8', errors='replace') as f:
                    output = f.read()
                with open(tmp_err.name, 'r', encoding='utf-8', errors='replace') as f:
                    error = f.read()

            # Afficher les résultats
            if process.returncode != 0:
                st.write("Erreur :", error.strip())  # Afficher les erreurs
            else:
                st.write(output.strip())  # Afficher la sortie standard
