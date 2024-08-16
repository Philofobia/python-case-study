import os
from dotenv import load_dotenv
from openai import OpenAI
from chromadb.utils import embedding_functions
from chromadb import chromadb

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
    response = openai_client.embeddings.create(input=text, model= "text-embedding-3-small")
    embedding = response.data[0].embedding
    return embedding

# Initialize chroma client with persistence
chroma_client = chromadb.PersistentClient(path=DATABASE_PATH)
collection_name = "documents_collection"
collection = chroma_client.get_or_create_collection(name=collection_name)# Initialize OpenAI client
openai_client = OpenAI(api_key=openai_key)
print("OpenAI client initialized", openai_client)

documents = load_documents_from_directory(DIRECTORY_PATH)
document_chunks = split_documents(documents)

# Upsert documents into collection
for document in document_chunks:
    document["embedding"] = get_openai_embedding(document["text"])
    collection.upsert(ids=[document["id"]],documents=[document["text"]], embeddings=[document["embedding"]])

# def query documents
def query_documents(query, num_results=2):
    # query_embedding = get_openai_embedding(query)
    results = collection.query(query_texts=query, num_results=num_results)

    #extract relevant chunks
    # relevant chunks is a list of documents that are relevant to the query
    # [documents for sublist in results["documents"] for documents in sublist] is used to flatten the list
    # which means that we are getting a list of documents instead of a list of lists of documents
    relevant_chunks = [documents for sublist in results["documents"] for documents in sublist]
    print("Return relevant chunks")
    for idx, document in enumerate(results["documents"]):
        doc_id = results["ids"][0][idx]
        distance = results["distances"][0][idx]
        print(f"Document ID: {doc_id}, Distance: {distance}")
    return relevant_chunks

# response = openai_client.chat.completions.create(
#    model="gpt-3.5-turbo",
#   messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "What is the capital of the United States?"},
#    ],
# )