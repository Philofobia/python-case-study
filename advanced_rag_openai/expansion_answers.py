from helper_utils import word_wrap
import os
from pypdf import PdfReader
from openai import OpenAI
from dotenv import load_dotenv
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter
)
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from chromadb import PersistentClient

DIRECTORY_PATH = os.path.join(os.path.dirname(__file__), "documents")
DATABASE_PATH = os.path.join(os.path.dirname(__file__), "database")

# Load the environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Extract the text from the PDF
reader = PdfReader(f"{DIRECTORY_PATH}/SecondBrain.pdf")
pdf_texts = [p.extract_text().strip() for p in reader.pages]
# filter out empty strings
pdf_texts = [text for text in pdf_texts if text]



# split text in chunks
character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0
)
character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))

# split text in tokens
token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0, tokens_per_chunk=256
)
token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)


embedding_function = SentenceTransformerEmbeddingFunction()
print("Embedding function: ", embedding_function)

# Initialize chromadb client
chromadb_client = PersistentClient(path=DATABASE_PATH)

chroma_collection = chromadb_client.get_or_create_collection(
    "pdf-collection", embedding_function=embedding_function
)

# extract the embeddings of the token_split_texts
ids = [str(i) for i in range(len(token_split_texts))]
chroma_collection.add(ids=ids, documents=token_split_texts)
chroma_collection.count()


query = "What are steps to learn how to take notes?"

results = chroma_collection.query(query_texts=[query], n_results=5)
retrieved_documents = results["documents"][0]

print("Retrieved documents: ", retrieved_documents)