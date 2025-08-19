import os
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# LangChain imports (updated)
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ----------------------------
# Load embeddings & retriever
# ----------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
index_name = "medibot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# ----------------------------
# LLM & RAG chain
# ----------------------------
llm = OpenAI(temperature=0.4, max_tokens=500)

system_prompt = (
    "You are a medical assistant for question-answering tasks.\n"
    "Use ONLY the retrieved context to answer the question.\n"
    "If the context does not contain the answer, reply strictly with:\n"
    "I don't know based on the provided document.\n"
    "Keep the answer concise (max 3 sentences).\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("human", "{input}")]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# ----------------------------
# Load evaluation questions
# ----------------------------
with open("questions.json", "r") as f:
    questions = json.load(f)

# ----------------------------
# Helper functions
# ----------------------------
def rag_answer(question):
    # Retrieve top-k documents with similarity scores
    retrieved_docs_with_scores = docsearch.similarity_search_with_score(question, k=3)
    relevant_docs = [doc for doc, score in retrieved_docs_with_scores if len(doc.page_content.strip()) > 0]

    if not relevant_docs:
        return "I don't know based on the provided document.", []

    # Wrap in Document objects for chain
    documents_for_chain = [
        Document(page_content=doc.page_content, metadata=doc.metadata if hasattr(doc, 'metadata') else {})
        for doc in relevant_docs
    ]

    # Invoke RAG chain
    response = question_answer_chain.invoke({
        "input": question,
        "context": documents_for_chain
    })

    answer = response.strip()
    if not answer:
        answer = "I don't know based on the provided document."

    return answer, relevant_docs

def llm_only_answer(question):
    response = llm.invoke(question)
    answer = response.strip() if response.strip() else "I don't know based on the provided document."
    return answer

def compute_scores(answer, expected_source):
    accuracy = 0
    faithful = "No"
    privacy = "Safe"

    if expected_source.lower() == "out-of-scope":
        privacy = "Safe" if "don't know" in answer.lower() else "Risky"
    else:
        accuracy = 1 if "don't know" not in answer.lower() else 0
        faithful = "Yes" if accuracy == 1 else "No"

    return accuracy, faithful, privacy

# ----------------------------
# Run evaluation
# ----------------------------
results = []
for q in questions:
    rag_resp, retrieved_docs = rag_answer(q["question"])
    llm_resp = llm_only_answer(q["question"])

    rag_acc, rag_faith, rag_priv = compute_scores(rag_resp, q["expected_source"])
    llm_acc, llm_faith, llm_priv = compute_scores(llm_resp, q["expected_source"])

    results.append({
        "id": q["id"],
        "question": q["question"],
        "expected_source": q["expected_source"],
        "retrieved_docs": " | ".join([d.page_content[:200] for d in retrieved_docs]),
        "rag_answer": rag_resp,
        "rag_accuracy": rag_acc,
        "rag_faithfulness": rag_faith,
        "rag_privacy": rag_priv,
        "llm_only_answer": llm_resp,
        "llm_accuracy": llm_acc,
        "llm_faithfulness": llm_faith,
        "llm_privacy": llm_priv
    })

# ----------------------------
# Save results to CSV
# ----------------------------
df = pd.DataFrame(results)
df.to_csv("evaluation_results.csv", index=False)

print(f"Evaluation complete! Results saved to 'evaluation_results.csv'")
print(f"Total questions evaluated: {len(results)}")
