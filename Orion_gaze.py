import os
import tempfile
import logging
import warnings
import time
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.indexes import VectorstoreIndexCreator
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAI
from langchain_community.vectorstores import FAISS
# from langchain.schema import Document

# ------------------------- Setup -------------------------
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
load_dotenv()

st.set_page_config(page_title="📚 Orion's Gaze - RAG Chatbot", layout="wide")

# ------------------------- Helper Functions -------------------------
def load_document(uploaded_file):
    suffix = uploaded_file.name.lower().split('.')[-1]
    if suffix != "pdf":
        st.error("❌ Only PDF files are supported.")
        return None

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    os.remove(tmp_path)
    return docs


# def create_vectorstore(docs):
#     embed_model = HuggingFaceEmbeddings(
#         model_name="thenlper/gte-small",
#         model_kwargs={"device": "cpu"}
#     )
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)
#     index = VectorstoreIndexCreator(
#         embedding=embed_model,
#         text_splitter=text_splitter
#     ).from_documents(docs)
#     return index.vectorstore
def create_vectorstore(docs):
    """Splits, embeds, and stores documents in a FAISS vectorstore."""
    embed_model = HuggingFaceEmbeddings(
        model_name="thenlper/gte-small",
        model_kwargs={"device": "cpu"}
    )

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)
    split_docs = text_splitter.split_documents(docs)

    # Create a FAISS index from document chunks
    vectorstore = FAISS.from_documents(split_docs, embed_model)
    return vectorstore
# ------------------------- Session Initialization -------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstores" not in st.session_state:
    st.session_state.vectorstores = {}
if "selected_docs" not in st.session_state:
    st.session_state.selected_docs = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "llm_fallback" not in st.session_state:
    st.session_state.llm_fallback = False
# ------------------------- Sidebar ui -------------------------
# The <style> block must be inside a string
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] h1 {
        font-size: 35px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# Place this function with your other helper functions, or right before the sidebar code.
def clear_all_state():
    """Clears all relevant session state variables."""
    st.session_state.vectorstores = {}
    st.session_state.selected_docs = []
    st.session_state.chat_history = []
    # --- CHANGE ---
    # Increment the key to force a widget replacement
    st.session_state.uploader_key += 1

# ------------------------- Sidebar -------------------------
with st.sidebar:
    st.title("🔭 Orion's Gaze")
    st.header("📂 Upload Documents")

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] h1 { font-size: 35px !important; }
        [data-testid="stFileUploaderFile"] { display: none; }
        </style>
        """,
        unsafe_allow_html=True,
    )
        # --- Custom accent color ---
    st.markdown(
        """
        <style>
        :root {
            --primary-color: #744ee6 !important; /* Change the orange to greenish-teal */
            --secondary-background-color: #744ee6 !important; /* Subtle green-tinted background */
        }

        /* Override Streamlit multiselect highlight */
        div[data-baseweb="select"] span {
            background-color: #9d72e8 !important; /* Selected item background */
            color: white !important;              /* Text color */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


    # --- CHANGE ---
    uploaded_files = st.file_uploader(
        "Upload PDF files only",
        accept_multiple_files=True,
        type=["pdf"],
        key=st.session_state.uploader_key # Use the dynamic key
    )

    if uploaded_files:
        processed_files_this_run = []
        for file in uploaded_files:
            if file.name not in st.session_state.vectorstores:
                with st.spinner(f"Processing {file.name}... please wait ⏳"):
                    docs = load_document(file)
                    if docs:
                        vstore = create_vectorstore(docs)
                        st.session_state.vectorstores[file.name] = vstore
                        processed_files_this_run.append(file.name)
        
        if processed_files_this_run:
            success_placeholder = st.empty()
            success_placeholder.success(f"✅ Processed: {', '.join(processed_files_this_run)}")
            time.sleep(3)  # show for 3 seconds
            success_placeholder.empty()  # clear it


    if st.session_state.vectorstores:
        st.session_state.selected_docs = st.multiselect(
            "Select processed documents for retrieval",
            options=list(st.session_state.vectorstores.keys()),
            default=st.session_state.selected_docs
        )

        st.markdown("---")
        # --- CORRECTED BUTTON LOGIC ---
        st.button(
            "🗑️ Clear All Documents & Chat",
            on_click=clear_all_state # Use the callback here
        )
        # --- End of Corrected Section ---

    else:
        st.info("📥 Upload PDF files to start building your knowledge base.")

    st.markdown("---")
    st.subheader("🧠 LLM Mode")
    st.checkbox(
        "💡 Use Gemini 2.5",
        key="llm_fallback",
        help="When enabled, Orion combines knowledge from your PDFs with its own understanding for better answers."
    )

    st.caption("🧠 Powered by Dileep + Mohan | Orion’s Gaze Edition")
# # ------------------------- Main Chat Section -------------------------
# st.title("💬 Orion’s Gaze - A hunter focusing intensely to find a specific target")
# st.write("Ask questions from your uploaded PDFs.")

# for msg in st.session_state.chat_history:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])

# if user_prompt := st.chat_input("Ask your question here..."):
#     st.session_state.chat_history.append({"role": "user", "content": user_prompt})
#     with st.chat_message("user"):
#         st.markdown(user_prompt)

#     if not st.session_state.selected_docs:
#         response = "⚠️ Please select at least one processed document from the sidebar."
#         st.session_state.chat_history.append({"role": "assistant", "content": response})
#         with st.chat_message("assistant"):
#             st.markdown(response)
#     else:
#         try:
#             api_key = os.getenv("GOOGLE_API_KEY")
#             if not api_key:
#                 raise ValueError("Missing GOOGLE_API_KEY in .env")

#             llm = GoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

#             all_docs = []
#             for name in st.session_state.selected_docs:
#                 retriever = st.session_state.vectorstores[name].as_retriever(search_kwargs={"k": 15})
#                 docs = retriever.get_relevant_documents(user_prompt)
#                 all_docs.extend(docs)

#             unique_docs = {doc.page_content: doc for doc in all_docs}.values()

#             with st.expander("🔍 Retrieved Context Preview"):
#                 for i, doc in enumerate(unique_docs):
#                     st.markdown(f"**Chunk {i+1}:** {doc.page_content[:300]}...")

#             context_text = "\n\n".join([doc.page_content for doc in unique_docs])
#             prompt = f"Answer the following question using ONLY the context below.\n\nContext:\n{context_text}\n\nQuestion: {user_prompt}"

#             full_response = ""
#             with st.chat_message("assistant"):
#                 placeholder = st.empty()
#                 for token in llm.stream(prompt):
#                     full_response += token
#                     placeholder.markdown(full_response + "▌")
#                     time.sleep(0.02)
#                 placeholder.markdown(full_response)

#             st.session_state.chat_history.append({"role": "assistant", "content": full_response})

#         except Exception as e:
#             err = f"❌ Error: {str(e)}"
#             st.session_state.chat_history.append({"role": "assistant", "content": err})
#             with st.chat_message("assistant"):
#                 st.markdown(err)

# # ------------------------- Footer -------------------------
# st.markdown("---")
# # st.caption("🚀 Orion’s Gaze | Streamed Gemini responses + optimized FAISS retrieval")

# ------------------------- Main Chat Section -------------------------
st.title("💬 Orion’s Gaze - A hunter focusing intensely to find a specific target")
st.write("Ask questions from your uploaded PDFs.")

# # --- 1. ADD CHECKBOX FOR LLM FALLBACK ---
# st.checkbox(
#     "Allow model to use general knowledge if answer is not in documents",
#     key="llm_fallback",
#     help="When checked, the AI can answer from its own knowledge if the uploaded PDFs don't contain the answer. When unchecked, it will only use the PDFs."
# )
# st.markdown("---")


for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_prompt := st.chat_input("Ask your question here..."):
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    if not st.session_state.selected_docs:
        # This part is for the general chat mode you added previously
        # We can enhance it slightly to let the user know the checkbox doesn't apply here
        with st.chat_message("assistant"):
             st.info("No documents selected. Responding in General Chat mode.", icon="🤖")
             try:
                api_key = os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError("Missing GOOGLE_API_KEY in .env")
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
                
                full_response = ""
                placeholder = st.empty()
                for token in llm.stream(user_prompt):
                    full_response += token.content
                    placeholder.markdown(full_response + "▌")
                    time.sleep(0.01)
                placeholder.markdown(full_response)
                st.session_state.chat_history.append({"role": "assistant", "content": full_response})

             except Exception as e:
                err = f"❌ Error: {str(e)}"
                st.session_state.chat_history.append({"role": "assistant", "content": err})
                st.markdown(err)

    else:
        # This is the RAG mode where the new logic applies
        try:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("Missing GOOGLE_API_KEY in .env")

            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

            all_docs = []
            for name in st.session_state.selected_docs:
                retriever = st.session_state.vectorstores[name].as_retriever(search_kwargs={"k": 15})
                docs = retriever.get_relevant_documents(user_prompt)
                all_docs.extend(docs)

            unique_docs = {doc.page_content: doc for doc in all_docs}.values()

            with st.expander("🔍 Retrieved Context Preview"):
                for i, doc in enumerate(unique_docs):
                    st.markdown(f"**Chunk {i+1}:** {doc.page_content[:300]}...")

            context_text = "\n\n".join([doc.page_content for doc in unique_docs])

           # --- 2. DYNAMICALLY CREATE THE PROMPT BASED ON THE CHECKBOX ---
            if st.session_state.llm_fallback:
                # ✅ HYBRID MODE PROMPT (Checkbox is ON)
                prompt = f"""
            You are a knowledgeable AI assistant called Orion.
            You have access to both the given context from uploaded PDFs and your own general knowledge.
            Use BOTH sources to generate the best possible answer.
            Prioritize facts from the context, but if helpful, expand or clarify using your broader understanding.

            At the start of your answer, include this line:
            "_ℹ️ This response is generated using both the uploaded PDFs and the model’s general knowledge._"

            ---
            Context from documents:
            {context_text}

            User Question:
            {user_prompt}
            ---
            Provide a clear, detailed answer:
            """
            else:
                # 🚫 STRICT MODE PROMPT (Checkbox is OFF)
                prompt = f"""
            You are a helpful assistant.
            Answer the following question using ONLY the context provided below.
            If the answer is not found in the context, reply:
            'The answer could not be found in the provided documents.'
            Do not use any external knowledge.

            ---
            Context:
            {context_text}

            User Question:
            {user_prompt}
            ---
            Provide a concise and factual answer:
            """

            # --- End of new logic ---

            full_response = ""
            with st.chat_message("assistant"):
                placeholder = st.empty()
                for token in llm.stream(prompt):
                    full_response += token.content
                    placeholder.markdown(full_response + "▌")
                    time.sleep(0.01)
                placeholder.markdown(full_response)

            st.session_state.chat_history.append({"role": "assistant", "content": full_response})

        except Exception as e:
            err = f"❌ Error: {str(e)}"
            st.session_state.chat_history.append({"role": "assistant", "content": err})
            with st.chat_message("assistant"):
                st.markdown(err)

# ------------------------- Footer -------------------------
st.markdown("---")
# st.caption("🚀 Orion’s Gaze | Streamed Gemini responses + optimized FAISS retrieval")