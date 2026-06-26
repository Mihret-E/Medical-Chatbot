from flask import Flask, render_template, jsonify, request
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
#from langchain_openai import ChatOpenAI
#from langchain_community.chains import RetrievalQA
#from langchain.chains import create_retrieval_chain
#from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
import certifi


os.environ["SSL_CERT_FILE"] = certifi.where()

from ollama import chat
from ollama import Client

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
#OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
#os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


embeddings = download_embeddings()

index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    embedding=embeddings,
    index_name=index_name
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


@app.route("/")
def index():
    return render_template('chat.html')






client = Client()

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("User input:", msg)

    docs = retriever.invoke(msg)
    context = "\n\n".join([d.page_content for d in docs])

    response = client.chat(
        model="llama3",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT + "\n\nContext:\n" + context
            },
            {
                "role": "user",
                "content": msg
            }
        ],
        options={
            "temperature": 0,
            "num_predict": 120
        }
    )

    answer = response["message"]["content"]

    print("Response:", answer)

    return str(answer)




if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)