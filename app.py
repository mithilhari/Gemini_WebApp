# app.py
import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
import re
import fitz
from docx import Document

# --- Configuration ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Google API Key not found. Please set it in a .env file.")
    st.stop()
try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    st.error(f"Error configuring Google API: {e}. Please check your API key.")
    st.stop()

# =======================================================
# CRITICAL FIX: st.set_page_config() MUST BE THE FIRST Streamlit command
st.set_page_config(page_title="LLM Document Summarizer & Concept Extractor", layout="centered")
# =======================================================

# --- Function to load and inject CSS ---
def inject_css(css_file_path):
    try:
        with open(css_file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"Error: CSS file not found at {css_file_path}. Please ensure it's in the 'static' folder.")

# Inject CSS from the static folder (now after set_page_config)
inject_css("static/style.css")


# --- Function to list available models ---
def list_available_models():
    """Lists available Gemini models and their capabilities."""
    st.sidebar.subheader("Available Models")
    try:
        supported_models = [
            m for m in genai.list_models() if "generateContent" in m.supported_generation_methods
        ]
        if supported_models:
            st.sidebar.markdown("Models supporting `generateContent`:")
            for model_info in supported_models:
                st.sidebar.markdown(f"- **`{model_info.name}`**")
        else:
            st.sidebar.warning("No models found supporting `generateContent`.")
            st.sidebar.info("If no models appear, check your API key and network connection.")

    except Exception as e:
        st.sidebar.error(f"Error listing models: {e}")
        st.sidebar.warning("Could not retrieve model list. This might indicate an API key issue or network problem.")

# --- LLM Setup ---
list_available_models()

try:
    model_name_to_use = 'models/gemini-2.0-flash' # Or 'models/gemini-2.0-flash' if available and desired
    model = genai.GenerativeModel(model_name_to_use)
except Exception as e:
    st.error(f"Failed to load LLM model '{model_name_to_use}': {e}. Please check the 'Available Models' in the sidebar and ensure the model name is correct and accessible.")
    st.stop()

# --- Helper Function for LLM Interaction ---
def get_llm_response(prompt_text):
    """Sends a prompt to the LLM and returns the response."""
    try:
        response = model.generate_content(prompt_text)
        return response.text
    except Exception as e:
        st.error(f"Error communicating with the LLM: {e}")
        st.info("Consider trying with a smaller document or adjusting the prompt.")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            st.code(f"API Error Details: {e.response.text}")
        return None

# --- Document Processing Helpers ---
def split_text_into_chunks(text, max_chunk_size=10000):
    chunks = []
    current_chunk = ""
    paragraphs = text.split('\n')
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) + 1 < max_chunk_size:
            current_chunk += paragraph + '\n'
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph + '\n'
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def read_txt(file):
    return file.read().decode("utf-8")

def read_pdf(file):
    text = ""
    try:
        pdf_document = fitz.open(stream=file.read(), filetype="pdf")
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
        pdf_document.close()
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        text = ""
    return text

def read_docx(file):
    text = ""
    try:
        document = Document(file)
        for paragraph in document.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        st.error(f"Error reading DOCX file: {e}")
        text = ""
    return text

def get_document_content(uploaded_file):
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension == "txt":
            return read_txt(uploaded_file)
        elif file_extension == "pdf":
            return read_pdf(uploaded_file)
        elif file_extension == "docx":
            return read_docx(uploaded_file)
        else:
            st.warning(f"Unsupported file type: .{file_extension}. Please upload a .txt, .pdf, or .docx file.")
            return ""
    return ""


# --- Streamlit UI ---
# Use a custom div to wrap content for Google-like centering and max-width
st.markdown("<div class='google-container'>", unsafe_allow_html=True)

st.title("📄 LLM Document Summarizer & Concept Extractor")
st.markdown("""
<p class='center-text description'>
Upload a document (.txt, .pdf, .docx) or paste text to get a concise summary and key concepts using Google's Gemini LLM.
</p>
<p class='center-text hint'>
<strong>Note:</strong> For very large documents, the summarization might be less precise or hit token limits.
</p>
""", unsafe_allow_html=True)


# Input method selection
st.markdown("<h3 class='center-text section-title'>Choose Input Method</h3>", unsafe_allow_html=True)
# Adjust radio button styling in CSS for a more compact, Google-like segment control
col_radio1, col_radio2, col_radio3 = st.columns([1,2,1])
with col_radio2: # Center the radio buttons
    input_method = st.radio("", ("Upload Document", "Paste Text"), key="input_method_radio", horizontal=True)


document_content = ""

if input_method == "Upload Document":
    st.markdown("<div class='input-card'>", unsafe_allow_html=True) # Card for input
    uploaded_file = st.file_uploader(
        "Upload a document (.txt, .pdf, .docx)",
        type=["txt", "pdf", "docx"],
        key="file_uploader",
        label_visibility="collapsed" # Hide default label to control it with markdown
    )
    if uploaded_file is not None:
        document_content = get_document_content(uploaded_file)
        if document_content:
            st.success("Document uploaded and read successfully!")
        else:
            st.error("Could not read content from the uploaded file.")
    st.markdown("</div>", unsafe_allow_html=True) # Close input card

elif input_method == "Paste Text":
    st.markdown("<div class='input-card'>", unsafe_allow_html=True) # Card for input
    pasted_text = st.text_area("Paste your text here:", height=250, key="paste_text_area",
                               label_visibility="collapsed", placeholder="Paste your document text here...")
    if pasted_text:
        document_content = pasted_text
    st.markdown("</div>", unsafe_allow_html=True) # Close input card


if document_content:
    st.markdown("<div class='results-card'>", unsafe_allow_html=True) # Card for results
    st.subheader("Document Preview (first 500 chars):")
    st.code(document_content[:500] + "..." if len(document_content) > 500 else document_content)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("Summarize and Extract Concepts", use_container_width=True): # Button in middle column
            if not document_content.strip():
                st.warning("Please upload a document or paste some text first.")
            else:
                with st.spinner("Processing document with LLM... This may take a moment."):
                    text_chunks = split_text_into_chunks(document_content)

                    full_summary_parts = []
                    all_concepts = set()

                    if len(text_chunks) > 1:
                        st.info(f"Document split into {len(text_chunks)} chunks for processing.")

                    for i, chunk in enumerate(text_chunks):
                        summary_prompt = f"""
                        Please provide a concise summary of the following text.
                        Also, extract up to 5 important key concepts or terms from this text,
                        and list them clearly at the end, prefixed with a bullet point.

                        Text:
                        ---
                        {chunk}
                        ---

                        Summary:
                        Key Concepts:
                        """
                        summary_and_concepts = get_llm_response(summary_prompt)

                        if summary_and_concepts:
                            full_summary_parts.append(summary_and_concepts)
                            concept_match = re.search(r"Key Concepts:\s*(.*)", summary_and_concepts, re.IGNORECASE | re.DOTALL)
                            if concept_match:
                                concepts_str = concept_match.group(1).strip()
                                current_concepts = [
                                    c.strip() for c in re.split(r'[,;•-]\s*|\n', concepts_str) if c.strip()
                                ]
                                all_concepts.update(current_concepts)
                        else:
                            st.warning(f"Could not process chunk {i+1}. Skipping this chunk.")

                    if full_summary_parts:
                        st.subheader("Summarization Results:")
                        st.markdown("---")

                        if len(text_chunks) > 1 and len(full_summary_parts) == len(text_chunks):
                            st.markdown("### Combined Summary (from chunks):")
                            combined_text_for_final_summary = "\n\n".join(full_summary_parts)
                            final_summary_prompt = f"""
                            Combine and refine the following partial summaries and extracted concepts into one cohesive, concise summary.
                            Also, present a final list of the most important unique key concepts identified across all parts, prefixed with a bullet point.
                            Ensure the summary flows well and captures the main essence.

                            Partial Summaries and Concepts:
                            ---
                            {combined_text_for_final_summary}
                            ---

                            Final Summary:
                            Final Key Concepts:
                            """
                            final_output = get_llm_response(final_summary_prompt)
                            if final_output:
                                st.write(final_output)
                            else:
                                st.error("Failed to generate a final combined summary. Displaying individual summaries:")
                                for i, summary in enumerate(full_summary_parts):
                                    st.markdown(f"**Chunk {i+1} Summary:**")
                                    st.write(summary)
                        else:
                            if len(full_summary_parts) == 1:
                                 st.write(full_summary_parts[0])
                            else:
                                 st.warning("Some chunks failed to process or combined summary could not be generated. Displaying individual summaries:")
                                 for i, summary in enumerate(full_summary_parts):
                                    st.markdown(f"**Chunk {i+1} Summary:**")
                                    st.write(summary)

                        st.markdown("---")
                        st.subheader("Extracted Key Concepts:")
                        if all_concepts:
                            st.markdown("<div class='concept-chips-container'>", unsafe_allow_html=True) # New container for chips
                            sorted_concepts = sorted(list(all_concepts))
                            for concept in sorted_concepts:
                                # Wrap each concept in a span with a custom class for styling
                                st.markdown(f"<span class='concept-chip'>{concept}</span>", unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True) # Close the container
                        else:
                            st.info("No distinct key concepts could be extracted from the processed parts.")

                    else:
                        st.error("Failed to generate any summary from the document. Please check the document content and API key.")
    st.markdown("</div>", unsafe_allow_html=True) # Close results card

st.markdown("</div>", unsafe_allow_html=True) # Close google-container
