import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFaceHub

# Cargar el documento
loader = TextLoader("ruta/a/tu/archivo.txt")
documentos = loader.load()

# Dividir el texto en fragmentos manejables
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
textos = text_splitter.split_documents(documentos)

# Crear embeddings usando un modelo de HuggingFace (no requiere API keys)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Crear base de datos vectorial
vectorstore = FAISS.from_documents(textos, embeddings)

# Configurar modelo de lenguaje (local)
llm = HuggingFaceHub(repo_id="google/flan-t5-base", 
                    model_kwargs={"temperature":0.1, "max_length":512})

# Crear cadena de preguntas y respuestas
cadena = load_qa_chain(llm, chain_type="stuff")

# Función para consultar información
def consultar_documento(pregunta):
    docs = vectorstore.similarity_search(pregunta)
    respuesta = cadena.run(input_documents=docs, question=pregunta)
    return respuesta

# Ejemplo de uso
respuesta = consultar_documento("¿Qué información específica quieres saber?")
print(respuesta)