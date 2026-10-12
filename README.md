# Cricket RAG Chatbot

## Project Overview
This project is a basic Retrieval-Augmented Generation (RAG) chatbot that answers questions about cricket using information stored in a local text file.

It retrieves relevant cricket information using semantic search and generates answers using a local language model.

## Technologies Used
- Python
- Sentence Transformers for text embeddings
- ChromaDB for vector storage and retrieval
- Ollama for local language model responses
- Llama 3.2 as the language model

## Project Structure
- `cricket_rag.py` — Main RAG pipeline
- `cricket.txt` — Cricket knowledge source
- `requirements.txt` — Python dependencies
- `.gitignore` — Files excluded from Git

## Prerequisites
- Python installed
- Ollama installed
- The Llama 3.2 model downloaded through Ollama



## Example Questions
- What are the three formats of cricket?
- What is a hat-trick in cricket?
- What is the difference between ODI and T20 cricket?

## How It Works
1. Reads cricket knowledge from `cricket.txt`.
2. Splits the text into smaller chunks.
3. Converts the chunks into embeddings.
4. Stores the embeddings in ChromaDB.
5. Retrieves relevant information for a user's question.
6. Uses Ollama and Llama 3.2 to generate an answer.



