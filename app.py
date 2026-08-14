import os
import base64
import html
from pathlib import Path

import streamlit as st
from google import genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="PDF AI Assistant",
    page_icon="📚",
    layout="wide"
)


# =========================================================
# PROJECT PATH
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

CSS_PATH = BASE_DIR / "style.css"
IMAGE_PATH = BASE_DIR / "background.png"


# =========================================================
# LOAD CSS
# =========================================================

try:

    with open(CSS_PATH, "r", encoding="utf-8") as f:
        css = f.read()

except FileNotFoundError:

    st.error("❌ style.css file not found.")
    st.stop()


# =========================================================
# LOAD BACKGROUND IMAGE
# =========================================================

try:

    with open(IMAGE_PATH, "rb") as img:
        encoded_image = base64.b64encode(
            img.read()
        ).decode("utf-8")

except FileNotFoundError:

    st.error(
        "❌ Background image not found. "
        "Check: assets/background.png"
    )
    st.stop()


# =========================================================
# APPLY BACKGROUND
# =========================================================

background_css = f"""
<style>

.stApp {{
    background-image: url("data:image/png;base64,{encoded_image}") !important;
    background-size: cover !important;
    background-position: center !important;
    background-attachment: fixed !important;
    background-repeat: no-repeat !important;
    min-height: 100vh !important;
}}

[data-testid="stAppViewContainer"] {{
    background: transparent !important;
}}

[data-testid="stHeader"] {{
    background: transparent !important;
}}

.main .block-container {{
    max-width: 1100px;
    padding-top: 35px;
    padding-bottom: 50px;
}}

</style>
"""

st.markdown(
    background_css + f"<style>{css}</style>",
    unsafe_allow_html=True
)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">📚 PDF AI Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload a PDF and ask questions directly from your document.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# GEMINI API
# =========================================================

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

if not GEMINI_API_KEY:

    st.error("❌ Gemini API key not found.")
    st.stop()

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


model = load_embedding_model()


# =========================================================
# CHROMADB
# =========================================================

client = chromadb.PersistentClient(
    path=str(BASE_DIR / "chroma_db")
)

collection = client.get_or_create_collection(
    name="pdf_documents"
)


# =========================================================
# UPLOAD SECTION
# =========================================================

st.markdown(
    '<div class="section-title">📄 Upload your document</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="upload-description">'
    'Upload a PDF document to start asking questions.'
    '</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
    label_visibility="collapsed"
)


# =========================================================
# PROCESS PDF
# =========================================================

if uploaded_file is not None:

    try:

        # =====================================================
        # PDF UPLOADED
        # =====================================================

        st.success(
            f"✅ PDF uploaded: {uploaded_file.name}"
        )


        # =====================================================
        # READ PDF
        # =====================================================

        reader = PdfReader(uploaded_file)

        pages = []

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            text = page.extract_text()

            if text:

                pages.append(
                    {
                        "page": page_number,
                        "text": text
                    }
                )


        st.write(
            f"📄 Pages extracted: {len(pages)}"
        )


        # =====================================================
        # CREATE CHUNKS
        # =====================================================

        chunks = []

        chunk_size = 1000
        overlap = 200

        for page in pages:

            text = page["text"]

            start = 0

            while start < len(text):

                end = start + chunk_size

                chunk_text = text[start:end]

                chunks.append(
                    {
                        "text": chunk_text,
                        "page": page["page"]
                    }
                )

                start += (
                    chunk_size - overlap
                )


        st.write(
            f"🧩 Chunks created: {len(chunks)}"
        )


        # =====================================================
        # CREATE EMBEDDINGS
        # =====================================================

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = model.encode(texts)


        # =====================================================
        # CLEAR OLD DOCUMENT
        # =====================================================

        if collection.count() > 0:

            old_ids = collection.get()["ids"]

            if old_ids:

                collection.delete(
                    ids=old_ids
                )


        # =====================================================
        # STORE DOCUMENT IN CHROMADB
        # =====================================================

        ids = [
            f"chunk_{i}"
            for i in range(len(chunks))
        ]

        metadatas = [
            {
                "page": chunk["page"],
                "filename": uploaded_file.name
            }
            for chunk in chunks
        ]

        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )


        st.success(
            "🧠 PDF processed successfully!"
        )


        # =====================================================
        # QUESTION SECTION
        # =====================================================

        st.markdown(
            '<div class="section-title">🔎 Ask a Question</div>',
            unsafe_allow_html=True
        )

        question = st.text_input(
            "Enter your question:",
            placeholder="e.g. What is the CTC?"
        )


        # =====================================================
        # SEARCH + ANSWER
        # =====================================================

        if question:

            # =================================================
            # QUESTION EMBEDDING
            # =================================================

            question_embedding = model.encode(
                [question]
            )


            # =================================================
            # SEARCH DOCUMENT
            # =================================================

            results = collection.query(
                query_embeddings=question_embedding.tolist(),
                n_results=3
            )

            documents = results["documents"][0]

            sources = results["metadatas"][0]


            # =================================================
            # CREATE CONTEXT
            # =================================================

            context = ""

            for i in range(len(documents)):

                context += f"""
Source {i + 1}

Page: {sources[i]["page"]}

Content:
{documents[i]}

-------------------------
"""


            # =================================================
            # GEMINI PROMPT
            # =================================================

            prompt = f"""
You are a helpful PDF AI Assistant.

Answer the user's question using ONLY
the information provided in the PDF context.

Do not use outside knowledge.

If the answer is not present in the PDF,
say:

"I could not find this information in the uploaded PDF."

Be accurate, concise and professional.

PDF Context:
{context}

User Question:
{question}
"""


            # =================================================
            # GENERATE ANSWER
            # =================================================

            response = gemini_client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt
            )


            # =================================================
            # AI ANSWER
            # =================================================

            answer_text = response.text

            # Escape HTML so Gemini response
            # cannot break our HTML layout
            safe_answer = html.escape(
                answer_text
            ).replace(
                "\n",
                "<br>"
            )


            # IMPORTANT:
            # HTML starts from column 1.
            # This prevents Streamlit from
            # treating it as a code block.

            answer_html = f"""
<div class="answer-section">
    <div class="section-title"></div>
    <div class="answer-box">
        {safe_answer}
    </div>
</div>
"""

            st.markdown(
                answer_html,
                unsafe_allow_html=True
            )

    except Exception as e:

        st.error(
            f"❌ Error: {e}"
        )
