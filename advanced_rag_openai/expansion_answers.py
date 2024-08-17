from helper_utils import word_wrap, project_embeddings
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
import umap

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


query = "How to start taking notes?"

results = chroma_collection.query(query_texts=[query], n_results=5)
retrieved_documents = results["documents"][0]

# Initialize OpenAI client
openai_client = OpenAI(api_key=openai_api_key)

def augment_query_generated(query, model="gpt-3.5-turbo"):
    prompt = """You are a student who is trying to learn how to take notes. 
    You have read a book on the topic and want to use those information to answer any question to a friend."""
    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": query}]

    reponse = openai_client.chat.completions.create(
        model=model,
        messages=messages, )
    content = reponse.choices[0].message["content"]
    return content

original_query = "What are the steps needed to start taking good notes?"
hypothetical_answer = augment_query_generated(original_query)

joint_query = f"{original_query} {hypothetical_answer}"
print(word_wrap(joint_query))

results = chroma_collection.query(
    query_texts=joint_query, n_results=5, include=["documents", "embeddings"]
)
retrieved_documents = results["documents"][0]

embeddings = chroma_collection.get(include=["embeddings"])["embeddings"]
umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)
projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)


retrieved_embeddings = results["embeddings"][0]
original_query_embedding = embedding_function([original_query])
augmented_query_embedding = embedding_function([joint_query])

projected_original_query_embedding = project_embeddings(
    original_query_embedding, umap_transform
)
projected_augmented_query_embedding = project_embeddings(
    augmented_query_embedding, umap_transform
)
projected_retrieved_embeddings = project_embeddings(
    retrieved_embeddings, umap_transform
)
