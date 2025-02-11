# 📚 Book Recommender Using LLM

## 🚀 Project Overview

This project is a \*\* Book Recommender System\*\* that provides book recommendations based on **semantic similarity, categories, and emotional tone**. It uses **Sentence Transformers for embeddings, ChromaDB for vector search, and Gradio for an interactive UI**.

## 🎯 Features

✅ **Semantic Search**: Finds books based on descriptions using **Sentence Transformers embeddings**.\
✅ **Category & Emotion Filtering**: Filter books by **category and emotional tone (joy, sadness, fear, etc.)**.\
✅ **Vector Search with ChromaDB**: Uses **similarity search** to retrieve the most relevant books.\
✅ **Interactive Dashboard**: Built with **Gradio**, allowing users to enter queries and receive recommendations.\
✅ **Error Handling**: Ensures robustness with checks for missing data and invalid inputs.

## 🛠️ Installation

Ensure Python 3.8+ is installed, then run:

```bash
pip install pandas numpy gradio langchain sentence-transformers chromadb dotenv
```

📂 Dataset
The recommender system is built using books_with_emotions.csv and tagged_description.txt.
The system extracts book metadata, descriptions, and emotional tone scores for enhanced recommendations.

🏗️ How It Works
1️⃣ Load Book Data:
Reads books_with_emotions.csv (title, authors, category, emotional scores).
Loads book descriptions from tagged_description.txt.
2️⃣ Create Embeddings & Vector Search:
Generates sentence embeddings using "intfloat/e5-large-v2" from Sentence Transformers.
Stores embeddings in ChromaDB for fast similarity search.
3️⃣ Retrieve Recommendations:
Performs semantic search on user queries.
Filters by category and emotional tone (joy, anger, fear, etc.).
Returns top 16 book recommendations.
4️⃣ Display in Gradio UI:
Uses Gradio Blocks UI with input fields and dropdowns.
Displays book covers and short descriptions.
🔍 Example Usage
1️⃣ Run the Application
python gradio-dashboard.py
2️⃣ Input a Book Description
For example:
A heartwarming story about friendship and resilience.
3️⃣ Select a Category and Emotion (Optional)
Category: Fiction, Mystery, Science Fiction, etc.
Tone: Happy, Suspenseful, Angry, etc.
4️⃣ View Recommendations
The system will display 16 books with their covers, titles, authors, and summaries.

📜 Code Overview
Load & Process Data

import pandas as pd

BOOKS_CSV = "books_with_emotions.csv"
books = pd.read_csv(BOOKS_CSV)
books["large_thumbnail"] = books["thumbnail"].fillna("cover-not-found.jpg") + "&fife=w800"
Load & Split Descriptions

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

TEXT_FILE = "tagged_description.txt"
raw_documents = TextLoader(TEXT_FILE).load()
text_splitter = CharacterTextSplitter(separator="\n")
documents = text_splitter.split_documents(raw_documents)
Create Embeddings & Store in ChromaDB

from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

model_name = "intfloat/e5-large-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': 'cuda'})

db_books = Chroma.from_documents(documents, embedding=embeddings)
Retrieve Book Recommendations

def retrieve_semantic_recommendations(query: str, category: str = None, tone: str = None) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k=50)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs if rec.page_content.strip().split()[0].isdigit()]
    book_recs = books[books["isbn13"].isin(books_list)].head(50)

    if category and category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(16)

    if tone:
        tone_mapping = {"Happy": "joy", "Sad": "sadness", "Angry": "anger", "Suspenseful": "fear"}
        if tone in tone_mapping:
            book_recs = book_recs.sort_values(by=tone_mapping[tone], ascending=False)

    return book_recs.head(16)
    
Deploy with Gradio

import gradio as gr

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# 📚 Semantic Book Recommender")

    user_query = gr.Textbox(label="Enter a book description:")
    category_dropdown = gr.Dropdown(choices=["All"] + list(books["simple_categories"].unique()), label="Select a category:")
    tone_dropdown = gr.Dropdown(choices=["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"], label="Select an emotional tone:")
    submit_button = gr.Button("Find Recommendations")

    output = gr.Gallery(label="Recommendations", columns=8, rows=2)

    submit_button.click(fn=recommend_books, inputs=[user_query, category_dropdown, tone_dropdown], outputs=output)

if __name__ == "__main__":
    dashboard.launch(share=True)
    
📊 Expected Output
Book Recommendations: 16 books with thumbnails, titles, and descriptions.
Interactive Search: Users enter a query and get instant suggestions.

📌 Results
Highly accurate recommendations based on semantic similarity & emotional tone.
Fast retrieval using ChromaDB vector search.
User-friendly interface with Gradio Blocks.
