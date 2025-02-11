import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
import gradio as gr

# Load environment variables
load_dotenv()

# Check if books_with_emotions.csv exists before reading
BOOKS_CSV = r"books_with_emotions.csv"
if not os.path.exists(BOOKS_CSV):
    raise FileNotFoundError(f"Error: {BOOKS_CSV} not found. Please check the file path.")

# Load book data
books = pd.read_csv(BOOKS_CSV)
books["large_thumbnail"] = books["thumbnail"].fillna("cover-not-found.jpg") + "&fife=w800"

# Ensure isbn13 column is properly formatted
if "isbn13" not in books.columns:
    raise KeyError("Error: Column 'isbn13' not found in books CSV.")

# Check if tagged_description.txt exists before loading
TEXT_FILE = "tagged_description.txt"
if not os.path.exists(TEXT_FILE):
    raise FileNotFoundError(f"Error: {TEXT_FILE} not found. Please check the file path.")

# Load book descriptions and create embeddings
raw_documents = TextLoader(TEXT_FILE).load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
model_name = "intfloat/e5-large-v2"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

db_books = Chroma.from_documents(documents, embedding=embeddings)



def retrieve_semantic_recommendations(
    query: str,
    category: str = None,
    tone: str = None,
    initial_top_k: int = 50,
    final_top_k: int = 16,
) -> pd.DataFrame:
    """Retrieve book recommendations based on semantic similarity, category, and tone."""

    try:
        recs = db_books.similarity_search(query, k=initial_top_k)
        books_list = [
            int(rec.page_content.strip('"').split()[0])
            for rec in recs
            if rec.page_content.strip().split()[0].isdigit()
        ]
        book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

        # Filter by category
        if category and category != "All":
            book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)

        # Sort based on emotion score
        tone_sorting = {
            "Happy": "joy",
            "Surprising": "surprise",
            "Angry": "anger",
            "Suspenseful": "fear",
            "Sad": "sadness",
        }
        if tone in tone_sorting:
            book_recs = book_recs.sort_values(by=tone_sorting[tone], ascending=False, inplace=False)

        return book_recs.head(final_top_k)


    
    except Exception as e:
        print(f"Error in retrieving recommendations: {e}")
        return pd.DataFrame()  # Return empty DataFrame if an error occurs


def recommend_books(query: str, category: str, tone: str):
    """Formats book recommendations for display in Gradio."""
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    
    if recommendations.empty:
        return [("cover-not-found.jpg", "No recommendations found. Try a different query.")]

    results = []
    for _, row in recommendations.iterrows():
        description = row.get("description", "No description available")
        truncated_description = " ".join(description.split()[:30]) + "..."

        authors = row.get("authors", "Unknown Author")
        authors_split = authors.split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = authors

        caption = f"{row.get('title', 'Unknown Title')} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))

    return results


# Gradio UI Setup
categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# 📚 Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(label="Enter a book description:", placeholder="e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(choices=categories, label="Select a category:", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Select an emotional tone:", value="All")
        submit_button = gr.Button("Find Recommendations")

    gr.Markdown("## Recommended Books")
    output = gr.Gallery(label="Recommendations", columns=8, rows=2)

    submit_button.click(fn=recommend_books, inputs=[user_query, category_dropdown, tone_dropdown], outputs=output)

if __name__ == "__main__":
    dashboard.launch(share=True)
