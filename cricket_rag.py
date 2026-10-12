
import chromadb
from sentence_transformers import SentenceTransformer
import ollama

# Read cricket knowledge
with open("cricket.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Split text into smaller parts
chunks = text.split("\n\n")
chunks = [c.strip() for c in chunks if c.strip()]

print("Cricket information loaded:", len(chunks), "chunks")

# Load the embedding model
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert text into vectors
embeddings = model.encode(chunks).tolist()

# Create the vector database
client = chromadb.PersistentClient(path="./cricket_db")

# Rebuild the collection each time this script runs
try:
    client.delete_collection("cricket_knowledge")
except Exception:
    pass

collection = client.create_collection("cricket_knowledge")

collection.add(
    ids=[str(i) for i in range(len(chunks))],
    documents=chunks,
    embeddings=embeddings
)

print("Cricket RAG is ready!")

# Ask questions
while True:
    question = input("\nAsk a cricket question (type exit to stop): ")

    if question.lower() == "exit":
        break

    # Find relevant cricket information
    question_vector = model.encode([question]).tolist()

    results = collection.query(
        query_embeddings=question_vector,
        n_results=min(3, len(chunks))
    )

    context = "\n\n".join(results["documents"][0])

    # Generate an answer using the local LLM
    prompt = f"""
Answer using only the cricket information below.
If the answer is missing, say you don't know.

Cricket information:
{context}

Question: {question}
Answer:
"""

    response = ollama.generate(
        model="llama3.2",
        prompt=prompt
    )

    print("\nAnswer:")
    print(response["response"])

    print("\nRetrieved information:")
    print(context)

print("Chatbot closed.")
