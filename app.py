import os
import base64
import hashlib
from pathlib import Path

import streamlit as st
from google import genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb


# =========================================================
# PAGE CONFIG
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
# LOAD BACKGROUND IMAGE
# =========================================================

if not IMAGE_PATH.exists():
    st.error("❌ background.png not found.")
    st.stop()

with open(IMAGE_PATH, "rb") as img:
    encoded_image = base64.b64encode(
        img.read()
    ).decode("utf-8")


# =========================================================
# LOAD CSS
# =========================================================

if not CSS_PATH.exists():
    st.error("❌ style.css not found.")
    st.stop()

with open(CSS_PATH, "r", encoding="utf-8") as f:
    css = f.read()


# =========================================================
# APPLY BACKGROUND
# =========================================================

background_css = f"""
<style>

.stApp {{
    background-image:
        url("data:image/png;base64,{encoded_image}") !important;

    background-size: cover !important;
    background-position: center center !important;
    background-repeat: no-repeat !important;
    background-attachment: fixed !important;
    min-height: 100vh !important;
}}

[data-testid="stAppViewContainer"] {{
    background: transparent !important;
}}

[data-testid="stHeader"] {{
    background: transparent !important;
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
    """
    <div class="main-title">
        📚 PDF AI Assistant
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Upload a PDF and ask questions directly from your document.
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# GEMINI API
# =========================================================

try:

    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

except Exception:

    st.error(
        "❌ GEMINI_API_KEY is not configured in Streamlit Secrets."
    )

    st.stop()


gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)


# =========================================================
# EMBEDDING MODEL
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

@st.cache_resource
def get_chroma_collection():

    # IMPORTANT:
    # Use in-memory ChromaDB instead of PersistentClient.
    # This avoids filesystem/database problems on Streamlit Cloud.

    client = chromadb.Client()

    collection = client.get_or_create_collection(
        name="pdf_documents"
    )

    return collection


collection = get_chroma_collection()


# =========================================================
# UPLOAD SECTION
# =========================================================

st.markdown(
    """
    <div class="section-title">
        📄 Upload your document
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="upload-description">
        Upload a PDF document to start asking questions.
    </div>
    """,
    unsafe_allow_html=True
)


uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"],
    accept_multiple_files=False,
    label_visibility="visible"
)


# =========================================================
# PROCESS PDF
# =========================================================

if uploaded_file is not None:

    try:

        # -------------------------------------------------
        # CREATE FILE HASH
        # -------------------------------------------------

        file_bytes = uploaded_file.getvalue()

        file_hash = hashlib.md5(
            file_bytes
        ).hexdigest()


        # -------------------------------------------------
        # PROCESS ONLY NEW PDF
        # -------------------------------------------------

        if st.session_state.get("file_hash") != file_hash:

            with st.spinner("🧠 Processing PDF..."):

                # -----------------------------------------
                # READ PDF
                # -----------------------------------------

                reader = PdfReader(
                    uploaded_file
                )

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


                if not pages:

                    st.error(
                        "❌ No readable text found in this PDF."
                    )

                    st.stop()


                # -----------------------------------------
                # CREATE CHUNKS
                # -----------------------------------------

                chunks = []

                chunk_size = 1000
                overlap = 200

                for page in pages:

                    text = page["text"]

                    start = 0

                    while start < len(text):

                        end = start + chunk_size

                        chunk_text = text[start:end]

                        if chunk_text.strip():

                            chunks.append(
                                {
                                    "text": chunk_text,
                                    "page": page["page"]
                                }
                            )

                        start += (
                            chunk_size - overlap
                        )


                # -----------------------------------------
                # CREATE EMBEDDINGS
                # -----------------------------------------

                texts = [
                    chunk["text"]
                    for chunk in chunks
                ]

                embeddings = model.encode(
                    texts,
                    show_progress_bar=False
                )


                # -----------------------------------------
                # CLEAR OLD DATA
                # -----------------------------------------

                old_data = collection.get()

                old_ids = old_data.get("ids", [])

                if old_ids:

                    collection.delete(
                        ids=old_ids
                    )


                # -----------------------------------------
                # STORE NEW DATA
                # -----------------------------------------

                ids = [
                    f"chunk_{i}"
                    for i in range(
                        len(chunks)
                    )
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


                # -----------------------------------------
                # SAVE SESSION DATA
                # -----------------------------------------

                st.session_state["file_hash"] = file_hash

                st.session_state["filename"] = uploaded_file.name

                st.session_state["pdf_ready"] = True

                st.session_state["pages"] = len(pages)


            st.success(
                f"✅ {uploaded_file.name} processed successfully!"
            )


        else:

            st.success(
                f"✅ {st.session_state['filename']} is ready!"
            )


        # =================================================
        # QUESTION SECTION
        # =================================================

        st.markdown(
            """
            <div class="section-title question-title">
                🔎 Ask a Question
            </div>
            """,
            unsafe_allow_html=True
        )


        question = st.text_input(
            "Enter your question:",
            placeholder="e.g. What is the CTC?"
        )


        # =================================================
        # SEARCH + ANSWER
        # =================================================

        if question:

            with st.spinner("🤖 Finding answer..."):

                # -----------------------------------------
                # QUESTION EMBEDDING
                # -----------------------------------------

                question_embedding = model.encode(
                    [question]
                )


                # -----------------------------------------
                # SEARCH DOCUMENT
                # -----------------------------------------

                results = collection.query(

                    query_embeddings=
                    question_embedding.tolist(),

                    n_results=3
                )


                documents = results["documents"][0]

                sources = results["metadatas"][0]


                # -----------------------------------------
                # CREATE CONTEXT
                # -----------------------------------------

                context = ""

                for i in range(
                    len(documents)
                ):

                    context += f"""
Source {i + 1}

Page: {sources[i]["page"]}

Content:
{documents[i]}

-------------------------
"""


                # -----------------------------------------
                # GEMINI PROMPT
                # -----------------------------------------

                prompt = f"""
You are a PDF question-answering assistant.

Answer the user's question using ONLY the information
available in the provided PDF context.

Do not use outside knowledge.

IMPORTANT:
Give ONLY the direct answer.

Do not start with phrases such as:
"Based on the provided PDF context"
"According to the PDF"
"The document states"
"Based on the information provided"

Do not explain unnecessarily.

If the user asks for:
- a name → give only the name
- a salary/CTC → give only the salary or CTC
- a date → give only the date
- a number → give only the number
- a company → give only the company name

If the answer is not present in the PDF, reply exactly:

I could not find this information in the uploaded PDF.

PDF CONTEXT:

{context}

USER QUESTION:

{question}
"""


                # -----------------------------------------
                # GENERATE ANSWER
                # -----------------------------------------

                response = gemini_client.models.generate_content(

                    model="gemini-3.5-flash-lite",

                    contents=prompt
                )


                answer_text = response.text.strip()


            # =================================================
            # DISPLAY ANSWER
            # =================================================

            st.markdown(
                f"""
                <div class="answer-section">

                    <div class="section-title">
                        🤖 AI Answer
                    </div>

                    <div class="answer-box">
                        {answer_text}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


    except Exception as e:

        st.error(
            f"❌ Error while processing PDF: {str(e)}"
        )
