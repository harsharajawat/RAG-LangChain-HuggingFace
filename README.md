# RAG Project with LangChain and HuggingFace

This repository contains a beginner-level implementation of Retrieval-Augmented Generation (RAG) using LangChain and HuggingFace libraries. The code is based on the concepts and examples from *Unlocking Data with Generative AI and RAG* by Keith Bourne.

## Project Overview

RAG is a powerful technique that combines document retrieval with generative AI models to produce contextually enriched answers. This project demonstrates how to:

- Load and parse web documents
- Split text into semantic chunks for embeddings
- Create and query a vector store using Chroma
- Use HuggingFace conversational language models for generating answers
- Chain components together with LangChain pipelines for end-to-end response generation

## Code Description

- The code loads content from a web URL (`https://kbourne.github.io/chapter1.html`) with BeautifulSoup filtering.
- It embeds the semantic text chunks using the `sentence-transformers/all-MiniLM-L6-v2` model.
- The embedded documents are indexed using the Chroma vector database.
- The HuggingFace conversational model `"google/gemma-2-2b-it"` is used as the generative LLM.
- LangChain's pipeline creates a retrieval-augmented generation chain that answers user queries based on the indexed documents.

## Usage

1. Clone this repository  
2. Create a `.env` file with your HuggingFace API key like:  
    ```
    HUGGINGFACEHUB_API_TOKEN=your_token_here
    ```
3. Install required dependencies (e.g., using pip):  
    ```
    pip install -r requirements.txt
    ```
4. Run the RAG code:  
    ```
    python RAG_Code.py
    ```
5. The example query `"What are the advantages of using RAG?"` will be processed and the answer printed.

## Notes

- Do not commit your `.env` file or API keys to the repository.
- This is a learning project; you can extend and tweak it as you explore generative AI and RAG more deeply.
- For more detailed explanations, refer to Keith Bourne’s *Unlocking Data with Generative AI and RAG* book.

## License

This project is for learning and personal use only. Please respect the original author’s work when sharing or adapting.

---

Feel free to reach out if you have questions or suggestions about this project!
