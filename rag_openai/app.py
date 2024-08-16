import os
from dotenv import load_dotenv
from openai import OpenAI
from chromadb.utils import embedding_functions
from chromadb import PersistentClient

# GLOBALS
DIRECTORY_PATH = os.path.join(os.path.dirname(__file__), "documents")
DATABASE_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")

# Load environment variables
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
openai_client = OpenAI(api_key=openai_key)
print("OpenAI client initialized", openai_client)

# function to load documents from a specific directory
def load_documents_from_directory(directory_path):
    print("Loading documents from directory: ", directory_path)
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(os.path.join(directory_path, filename), "r") as file:
                documents.append({ "id": filename, "text": file.read() })
    return documents

# split documents into chunks 
def split_documents(documents, chunk_size=1000, chunk_overlap=20):
    chunks = []
    for document in documents:
        text = document["text"]
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunk = text[i:i + chunk_size]
            chunks.append({ "id": document["id"], "text": chunk })
    return chunks

# Define embedding function for OpenAI
def get_openai_embedding(text):
    response = openai_client.embeddings.create(input=text, model="text-embedding-ada-002")
    embedding = response.data[0].embedding
    return embedding

# Initialize chroma client with persistence
chroma_client = PersistentClient(path=DATABASE_PATH)
collection_name = "documents_collection"
openai_client = OpenAI(api_key=openai_key)
collection = chroma_client.get_or_create_collection(name=collection_name)# Initialize OpenAI client
print("OpenAI client initialized", openai_client)

documents = load_documents_from_directory(DIRECTORY_PATH)
document_chunks = split_documents(documents)

# Upsert documents into collection
for document in document_chunks:
    document["embedding"] = get_openai_embedding(document["text"])
    print("Upserting document: ", document["id"])
    print("document length: ", len(document["embedding"]))
    collection.upsert(ids=[document["id"]],documents=[document["text"]], embeddings=[document["embedding"]])
    print("Document upserted")

# def query documents
def query_documents(query, num_results=2):
    embedded_query = get_openai_embedding(query)
    results = collection.query(query_embeddings=[embedded_query], n_results=num_results)
    print("Query results: ", results)

    #extract relevant chunks
    # relevant chunks is a list of documents that are relevant to the query
    # doc is a dictionary with keys "id" and "text"
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    print("Return relevant chunks")
    for idx, document in enumerate(results["documents"]):
        doc_id = results["ids"][0][idx]
        distance = results["distances"][0][idx]
        print(f"Document ID: {doc_id}, Distance: {distance}")
    return relevant_chunks

# Generate response from openai
def generate_response(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question answering.\n"
        "You are given a question and a list of relevant documents.\n"
        "You need to provide an answer to the question based on the given documents.\n"
        "If you cannot find the answer in the documents, you can say 'I don't know'.\n"
        "\nContext:\n" + context + "\n\nQuestion: " + question
    )
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": prompt}, {"role": "user", "content": question}]
        )
    return response.choices[0].message


# query example
question = ""
relevant_chunks = query_documents(question)
answer = generate_response(question, relevant_chunks)

print("Answer:", answer)
