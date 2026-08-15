import base64
import html
from pathlib import Path

import streamlit as st
from google import genai
from pypdf import PdfReader


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="PDF AI Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

CSS_FILE = BASE_DIR / "style.css"
BACKGROUND_FILE = BASE_DIR / "background.png"


# =========================================================
# LOAD BACKGROUND IMAGE
# =========================================================

if not BACKGROUND_FILE.exists():
    st.error("❌ background.png not found.")
    st.stop()

with open(BACKGROUND_FILE, "rb") as f:
    background_base64 = base64.b64encode(
        f.read()
    ).decode()


# =========================================================
# LOAD CSS
# =========================================================

if not CSS_FILE.exists():
    st.error("❌ style.css not found.")
    st.stop()

with open(CSS_FILE, "r", encoding="utf-8") as f:
    css = f.read()


# Replace placeholder from CSS
css = css.replace(
    "__BACKGROUND_IMAGE__",
    f"data:image/png;base64,{background_base64}"
)


st.markdown(
    f"<style>{css}</style>",
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


client = genai.Client(
    api_key=GEMINI_API_KEY
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
    "Upload PDF",
    type=["pdf"],
    label_visibility="collapsed"
)


# =========================================================
# PROCESS PDF
# =========================================================

if uploaded_file is not None:

    try:

        # -------------------------------------------------
        # READ PDF
        # -------------------------------------------------

        reader = PdfReader(uploaded_file)

        pages_text = []

        for page in reader.pages:

            text = page.extract_text()

            if text:
                pages_text.append(text)


        pdf_text = "\n\n".join(pages_text)


        # -------------------------------------------------
        # CHECK PDF
        # -------------------------------------------------

        if not pdf_text.strip():

            st.error(
                "❌ Text could not be extracted from this PDF."
            )

            st.stop()


        # -------------------------------------------------
        # LIMIT CONTEXT
        # -------------------------------------------------

        MAX_TEXT_LENGTH = 60000

        pdf_context = pdf_text[:MAX_TEXT_LENGTH]


        # -------------------------------------------------
        # SUCCESS
        # -------------------------------------------------

        st.success(
            f"✅ {uploaded_file.name} uploaded successfully."
        )


        # =================================================
        # QUESTION
        # =================================================

        st.markdown(
            """
            <div class="section-title">
                🔎 Ask a Question
            </div>
            """,
            unsafe_allow_html=True
        )


        question = st.text_input(
            "Your question",
            placeholder="e.g. What is the CTC?",
            label_visibility="collapsed"
        )


        # =================================================
        # ANSWER
        # =================================================

        if question.strip():

            with st.spinner("🤖 Finding the answer..."):

                prompt = f"""
You are a PDF question-answering assistant.

Answer the user's question ONLY using the information
contained in the uploaded PDF.

IMPORTANT:

- Give only the direct answer.
- Do not say "Based on the PDF context".
- Do not say "According to the PDF".
- Do not explain your reasoning.
- Do not mention the source.
- Do not mention page numbers unless explicitly asked.
- Keep the answer short and precise.
- If the question asks for a person's name, return only the name.
- If the question asks for salary or CTC, return only the salary/CTC.
- If the question asks for an email, return only the email.
- If the information is not present, say:
  "I could not find this information in the uploaded PDF."

PDF CONTENT:

{pdf_context}

USER QUESTION:

{question}
"""


                response = client.models.generate_content(
                    model="gemini-3.5-flash",
                    contents=prompt
                )


                answer = response.text.strip()


            # -------------------------------------------------
            # CLEAN ANSWER
            # -------------------------------------------------

            prefixes = [
                "Based on the provided PDF context,",
                "Based on the PDF context,",
                "According to the PDF,",
                "According to the provided PDF,"
            ]

            for prefix in prefixes:

                if answer.lower().startswith(
                    prefix.lower()
                ):

                    answer = answer[len(prefix):].strip()


            # -------------------------------------------------
            # SAFE HTML
            # -------------------------------------------------

            answer = html.escape(answer)

            answer = answer.replace(
                "\n",
                "<br>"
            )


            # -------------------------------------------------
            # DISPLAY
            # -------------------------------------------------

            st.markdown(
                f"""
                <div class="answer-section">
                <div class="section-title">
                🤖 AI Answer
                </div>
                <div class="answer-box">
                {answer}
                </div>
                </div>
                """,
                unsafe_allow_html=True
                )


    except Exception as e:
        st.error(f"❌ Error: {type(e).__name__}")
        st.error(str(e))

        print(
            "PDF Assistant Error:",
            repr(e)
        )
