import base64
import html
from pathlib import Path

import streamlit as st
from google import genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="PDF AI Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
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
    st.error("❌ background.png file not found.")
    st.stop()


# =========================================================
# APPLY CSS + BACKGROUND
# =========================================================

background_css = f"""
<style>

.stApp {{
    background-image: url("data:image/png;base64,{encoded_image}") !important;
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
    background_css + "<style>" + css + "</style>",
    unsafe_allow_html=True
)


# =========================================================
# HEADER
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
    st.error("❌ Gemini API key is not configured.")
    st.stop()


if not GEMINI_API_KEY:
    st.error("❌ Gemini API key is empty.")
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
# SESSION STATE
# =========================================================

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = ""

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if "file_id" not in st.session_state:
    st.session_state.file_id = None


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
    "Choose a PDF file",
    type=["pdf"],
    label_visibility="collapsed"
)


# =========================================================
# PROCESS PDF
# =========================================================

if uploaded_file is not None:

    # -----------------------------------------------------
    # Create unique ID for uploaded file
    # -----------------------------------------------------

    file_bytes = uploaded_file.getvalue()

    current_file_id = (
        uploaded_file.name,
        len(file_bytes)
    )


    # -----------------------------------------------------
    # Process only when a NEW file is uploaded
    # -----------------------------------------------------

    if current_file_id != st.session_state.file_id:

        try:

            with st.spinner("📖 Reading your PDF..."):

                # =================================================
                # READ PDF
                # =================================================

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
                        "❌ No readable text was found in this PDF."
                    )

                    st.stop()


                # =================================================
                # CREATE CHUNKS
                # =================================================

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


                if not chunks:

                    st.error(
                        "❌ Could not create text chunks from this PDF."
                    )

                    st.stop()


                # =================================================
                # CREATE EMBEDDINGS
                # =================================================

                texts = [
                    chunk["text"]
                    for chunk in chunks
                ]

                embeddings = model.encode(
                    texts,
                    normalize_embeddings=True
                )


                # =================================================
                # SAVE IN SESSION STATE
                # =================================================

                st.session_state.chunks = chunks

                st.session_state.embeddings = embeddings

                st.session_state.pdf_name = (
                    uploaded_file.name
                )

                st.session_state.file_id = (
                    current_file_id
                )

                st.session_state.pdf_processed = True


            # =================================================
            # SUCCESS
            # =================================================

            st.success(
                f"✅ {uploaded_file.name} processed successfully!"
            )


        except Exception as e:

            st.error(
                "❌ Unable to process this PDF."
            )

            st.caption(
                f"Error: {str(e)}"
            )

            st.stop()


    else:

        # Existing processed file
        st.success(
            f"✅ {st.session_state.pdf_name} is ready!"
        )


# =========================================================
# QUESTION SECTION
# =========================================================

if st.session_state.pdf_processed:

    st.markdown(
        """
        <div class="question-section">
            <div class="section-title">
                🔎 Ask a Question
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


    question = st.text_input(
        "Enter your question:",
        placeholder="e.g. What is the CTC?",
        label_visibility="collapsed"
    )


    # =====================================================
    # SEARCH + ANSWER
    # =====================================================

    if question:

        try:

            with st.spinner("🤖 Finding the answer..."):

                # =============================================
                # QUESTION EMBEDDING
                # =============================================

                question_embedding = model.encode(
                    [question],
                    normalize_embeddings=True
                )[0]


                # =============================================
                # COSINE SIMILARITY
                # =============================================

                scores = (
                    st.session_state.embeddings
                    @ question_embedding
                )


                # =============================================
                # TOP RESULTS
                # =============================================

                top_k = min(
                    3,
                    len(scores)
                )

                top_indices = scores.argsort()[
                    -top_k:
                ][::-1]


                selected_chunks = [
                    st.session_state.chunks[i]
                    for i in top_indices
                ]


                # =============================================
                # CREATE CONTEXT
                # =============================================

                context_parts = []

                for chunk in selected_chunks:

                    context_parts.append(
                        chunk["text"]
                    )


                context = "\n\n".join(
                    context_parts
                )


                # =============================================
                # GEMINI PROMPT
                # =============================================

                prompt = f"""
You are a PDF question-answering assistant.

Answer the user's question using ONLY
the information available in the uploaded PDF.

IMPORTANT RULES:

1. Do not use outside knowledge.
2. Do not mention the PDF.
3. Do not mention sources.
4. Do not say "Based on the provided PDF context".
5. Do not explain how you found the answer.
6. Give ONLY the direct answer.
7. Keep the answer short and precise.
8. If the requested information is not available,
   say exactly:

I could not find this information.

Examples:

Question: What is the name?
Answer: Gungun Mishra

Question: What is the CTC?
Answer: 6 LPA

Question: What is the email?
Answer: example@gmail.com

DOCUMENT CONTENT:
{context}

USER QUESTION:
{question}
"""


                # =============================================
                # GEMINI RESPONSE
                # =============================================

                response = gemini_client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=prompt
                )


                answer_text = response.text.strip()


            # =================================================
            # CLEAN ANSWER
            # =================================================

            safe_answer = html.escape(
                answer_text
            ).replace(
                "\n",
                "<br>"
            )


            # =================================================
            # ANSWER DISPLAY
            # =================================================

            answer_html = (
                '<div class="answer-section">'
                '<div class="section-title">🤖 AI Answer</div>'
                '<div class="answer-box">'
                f'{safe_answer}'
                '</div>'
                '</div>'
            )


            st.markdown(
                answer_html,
                unsafe_allow_html=True
            )


        except Exception as e:

            st.error(
                "❌ Something went wrong while generating the answer."
            )

            st.caption(
                f"Error: {str(e)}"
            )
