from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Initialize Flask
app = Flask(__name__)

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Load embeddings and Pinecone index
embeddings = download_hugging_face_embeddings()
index_name = "medibot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Retriever with top-k similarity
retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Define system prompt
system_prompt = (
    "You are a helpful medical assistant for question-answering tasks.\n"
    "Use ONLY the provided retrieved context to answer.\n"
    "If the context does not contain the answer, reply strictly with:\n"
    "'I don't know based on the provided document.'\n"
    "Keep the answer concise (max 3 sentences).\n\n"
    "{context}"
)

# Use ChatOpenAI instead of old OpenAI wrapper
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2, max_tokens=300)

# Create prompt + RAG chain
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]

    # Retrieve top-3 documents with similarity scores
    retrieved_docs_with_scores = docsearch.similarity_search_with_score(msg, k=3)

    # Filter out empty docs
    relevant_docs = [doc for doc, score in retrieved_docs_with_scores if doc.page_content.strip()]

    if not relevant_docs:
        return "I don’t know based on the provided document."

    # ✅ Pass documents directly (not string!)
    response = question_answer_chain.invoke({"input": msg, "context": relevant_docs})

    # response may be dict or string depending on chain
    final_answer = response.strip() if isinstance(response, str) else str(response)

    if not final_answer or "I don’t know" in final_answer:
        return "I don’t know based on the provided document."

    return final_answer



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
