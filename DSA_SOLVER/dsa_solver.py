# import os
# import re
# import streamlit as st
# from dotenv import load_dotenv
# from langchain.document_loaders import TextLoader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_groq import ChatGroq
# from langchain.schema import Document
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# import time
# import json

# # Page Configuration
# st.set_page_config(
#     page_title="LeetCode AI Assistant", 
#     page_icon="🚀", 
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for modern dark theme
# st.markdown("""
# <style>
# /* Import fonts */
# @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Fira+Code:wght@400;500&display=swap');

# /* Root variables */
# :root {
#     --primary-bg: #0a0e1a;
#     --secondary-bg: #1a1f35;
#     --accent-bg: #252a42;
#     --gradient-primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#     --gradient-secondary: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
#     --gradient-success: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
#     --text-primary: #ffffff;
#     --text-secondary: #a8b2d1;
#     --text-accent: #64ffda;
#     --border-color: rgba(255, 255, 255, 0.1);
#     --shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
#     --glow: 0 0 20px rgba(100, 255, 218, 0.3);
# }

# /* Global styles */
# .stApp {
#     background: var(--primary-bg);
#     font-family: 'Inter', sans-serif;
# }

# /* Hide Streamlit branding */
# #MainMenu {visibility: hidden;}
# footer {visibility: hidden;}
# header {visibility: hidden;}

# /* Sidebar styling */
# .css-1d391kg {
#     background: linear-gradient(180deg, var(--secondary-bg) 0%, var(--accent-bg) 100%);
#     border-right: 1px solid var(--border-color);
# }

# .sidebar .sidebar-content {
#     background: transparent;
# }

# /* Custom title */
# .main-title {
#     background: var(--gradient-primary);
#     -webkit-background-clip: text;
#     -webkit-text-fill-color: transparent;
#     background-clip: text;
#     font-size: 3rem;
#     font-weight: 700;
#     text-align: center;
#     margin: 2rem 0;
#     text-shadow: var(--glow);
#     animation: glow 2s ease-in-out infinite alternate;
# }

# @keyframes glow {
#     from { filter: brightness(1); }
#     to { filter: brightness(1.2); }
# }

# /* Chat container */
# .chat-container {
#     background: var(--secondary-bg);
#     border-radius: 20px;
#     padding: 2rem;
#     margin: 1rem 0;
#     border: 1px solid var(--border-color);
#     backdrop-filter: blur(10px);
#     box-shadow: var(--shadow);
#     position: relative;
#     overflow: hidden;
# }

# .chat-container::before {
#     content: '';
#     position: absolute;
#     top: 0;
#     left: 0;
#     right: 0;
#     height: 2px;
#     background: var(--gradient-primary);
# }

# /* Message bubbles */
# .user-message {
#     background: var(--gradient-primary);
#     color: white;
#     padding: 1rem 1.5rem;
#     border-radius: 20px 20px 5px 20px;
#     margin: 1rem 0;
#     max-width: 80%;
#     margin-left: auto;
#     box-shadow: var(--shadow);
#     position: relative;
#     animation: slideInRight 0.3s ease-out;
# }

# .assistant-message {
#     background: var(--accent-bg);
#     color: var(--text-primary);
#     padding: 1.5rem;
#     border-radius: 20px 20px 20px 5px;
#     margin: 1rem 0;
#     max-width: 85%;
#     border: 1px solid var(--border-color);
#     box-shadow: var(--shadow);
#     position: relative;
#     animation: slideInLeft 0.3s ease-out;
# }

# .vector-response {
#     border-left: 4px solid #4facfe;
#     background: rgba(79, 172, 254, 0.1);
# }

# .llm-response {
#     border-left: 4px solid #64ffda;
#     background: rgba(100, 255, 218, 0.1);
# }

# @keyframes slideInRight {
#     from { transform: translateX(100px); opacity: 0; }
#     to { transform: translateX(0); opacity: 1; }
# }

# @keyframes slideInLeft {
#     from { transform: translateX(-100px); opacity: 0; }
#     to { transform: translateX(0); opacity: 1; }
# }

# /* Input area */
# .input-container {
#     background: var(--secondary-bg);
#     border-radius: 15px;
#     padding: 1.5rem;
#     border: 1px solid var(--border-color);
#     backdrop-filter: blur(10px);
#     margin: 2rem 0;
# }

# /* Code blocks */
# .stCode {
#     background: var(--primary-bg) !important;
#     border: 1px solid var(--border-color);
#     border-radius: 10px;
#     font-family: 'Fira Code', monospace;
# }

# /* Buttons */
# .stButton button {
#     background: var(--gradient-primary);
#     color: white;
#     border: none;
#     border-radius: 25px;
#     padding: 0.75rem 2rem;
#     font-weight: 600;
#     transition: all 0.3s ease;
#     box-shadow: var(--shadow);
# }

# .stButton button:hover {
#     transform: translateY(-2px);
#     box-shadow: 0 12px 40px rgba(102, 126, 234, 0.4);
# }

# /* Selectbox */
# .stSelectbox > div > div {
#     background: var(--accent-bg);
#     border: 1px solid var(--border-color);
#     border-radius: 10px;
#     color: var(--text-primary);
# }

# /* Text area */
# .stTextArea textarea {
#     background: var(--accent-bg);
#     border: 1px solid var(--border-color);
#     border-radius: 10px;
#     color: var(--text-primary);
#     font-family: 'Inter', sans-serif;
# }

# /* Metrics */
# .metric-card {
#     background: var(--accent-bg);
#     padding: 1rem;
#     border-radius: 10px;
#     border: 1px solid var(--border-color);
#     text-align: center;
#     margin: 0.5rem;
# }

# /* Status indicators */
# .status-indicator {
#     display: inline-block;
#     width: 8px;
#     height: 8px;
#     border-radius: 50%;
#     margin-right: 8px;
#     animation: pulse 2s infinite;
# }

# .status-online { background: #00ff88; }
# .status-processing { background: #ffa500; }

# @keyframes pulse {
#     0% { opacity: 1; }
#     50% { opacity: 0.5; }
#     100% { opacity: 1; }
# }

# /* Scrollbar */
# ::-webkit-scrollbar {
#     width: 8px;
# }

# ::-webkit-scrollbar-track {
#     background: var(--primary-bg);
# }

# ::-webkit-scrollbar-thumb {
#     background: var(--gradient-primary);
#     border-radius: 4px;
# }

# /* Loading animation */
# .loading-dots {
#     display: inline-block;
# }

# .loading-dots::after {
#     content: '';
#     animation: dots 1.5s infinite;
# }

# @keyframes dots {
#     0%, 20% { content: '.'; }
#     40% { content: '..'; }
#     60%, 100% { content: '...'; }
# }

# /* Language badge */
# .language-badge {
#     background: var(--gradient-secondary);
#     color: white;
#     padding: 0.25rem 0.75rem;
#     border-radius: 20px;
#     font-size: 0.8rem;
#     font-weight: 500;
#     display: inline-block;
#     margin: 0.25rem 0;
# }

# /* Response type badges */
# .response-badge {
#     padding: 0.25rem 0.75rem;
#     border-radius: 15px;
#     font-size: 0.75rem;
#     font-weight: 500;
#     margin-bottom: 1rem;
#     display: inline-block;
# }

# .vector-badge {
#     background: rgba(79, 172, 254, 0.2);
#     color: #4facfe;
#     border: 1px solid #4facfe;
# }

# .llm-badge {
#     background: rgba(100, 255, 218, 0.2);
#     color: #64ffda;
#     border: 1px solid #64ffda;
# }

# .followup-badge {
#     background: rgba(255, 163, 0, 0.2);
#     color: #ffa300;
#     border: 1px solid #ffa300;
# }

# /* Hover effects */
# .hover-card {
#     transition: all 0.3s ease;
# }

# .hover-card:hover {
#     transform: translateY(-5px);
#     box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
# }
# </style>
# """, unsafe_allow_html=True)

# # Enhanced sidebar with modern design
# st.sidebar.markdown("""
# <div style="text-align: center; padding: 1rem;">
#     <h2 style="color: #64ffda; margin: 0;">⚙️ Settings</h2>
#     <div class="status-indicator status-online"></div>
#     <span style="color: #a8b2d1; font-size: 0.9rem;">AI Assistant Online</span>
# </div>
# """, unsafe_allow_html=True)

# # Language selection with enhanced styling
# language = st.sidebar.selectbox(
#     "🔤 Programming Language:",
#     ["Python", "C++", "Java", "JavaScript", "Go"],
#     index=0,
#     help="Choose your preferred language for code solutions"
# )

# st.sidebar.markdown(f'<div class="language-badge">{language}</div>', unsafe_allow_html=True)

# # Add some metrics
# st.sidebar.markdown("### 📊 Session Stats")
# col1, col2 = st.sidebar.columns(2)
# with col1:
#     st.markdown(f"""
#     <div class="metric-card">
#         <div style="color: #4facfe; font-size: 1.5rem; font-weight: bold;">
#             {len(st.session_state.get('chat_history', []))}
#         </div>
#         <div style="color: #a8b2d1; font-size: 0.8rem;">Questions</div>
#     </div>
#     """, unsafe_allow_html=True)

# with col2:
#     st.markdown(f"""
#     <div class="metric-card">
#         <div style="color: #64ffda; font-size: 1.5rem; font-weight: bold;">
#             {language}
#         </div>
#         <div style="color: #a8b2d1; font-size: 0.8rem;">Language</div>
#     </div>
#     """, unsafe_allow_html=True)

# # Main title with gradient effect
# st.markdown('<h1 class="main-title">🚀 LeetCode AI Assistant</h1>', unsafe_allow_html=True)

# # Subtitle
# st.markdown("""
# <div style="text-align: center; color: #a8b2d1; font-size: 1.1rem; margin-bottom: 2rem;">
#     Your intelligent companion for mastering Data Structures & Algorithms
# </div>
# """, unsafe_allow_html=True)

# # Load and split documents
# @st.cache_resource
# def load_docs():
#     try:
#         loader = TextLoader("DSA.md", encoding="utf-8")
#         documents = loader.load()
#         splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
#         return splitter.split_documents(documents)
#     except Exception as e:
#         st.error(f"📄 Could not load DSA.md file: {e}")
#         return []

# text = load_docs()

# # Embeddings
# @st.cache_resource
# def get_embeddings():
#     return HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2",
#         model_kwargs={"device": "cpu"},
#         encode_kwargs={"normalize_embeddings": False}
#     )

# try:
#     embeddings = get_embeddings()
#     st.sidebar.success("✅ Embeddings loaded")
# except Exception as e:
#     st.error(f"🔴 Embedding load error: {e}")
#     st.stop()

# # Vector store
# @st.cache_resource
# def get_vectorstore():
#     if text:
#         return FAISS.from_documents(text, embeddings)
#     return None

# vectorstore = get_vectorstore()
# if vectorstore:
#     retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
#     st.sidebar.success("✅ Vector DB ready")
# else:
#     st.sidebar.warning("⚠️ Vector DB not available")

# # Load GROQ API Key
# load_dotenv()
# try:
#     groq_api_key = st.secrets["GROQ_API_KEY"]
#     st.sidebar.success("✅ API Key loaded")
# except Exception:
#     st.sidebar.error("🔴 GROQ_API_KEY not found!")
#     groq_api_key = st.sidebar.text_input("Enter GROQ API Key:", type="password")
#     if not groq_api_key:
#         st.stop()

# # Language model
# model = ChatGroq(
#     groq_api_key=groq_api_key,
#     model="Llama3-8b-8192"
# )

# # Prompt template for RAG
# prompt = ChatPromptTemplate.from_messages([
#     ("system", f"You are an expert LeetCode assistant. Always provide solutions in {language} with clear explanations, time/space complexity analysis, and well-commented code."),
#     ("user", "Context: {context}\n\nQuestion: {question}")
# ])

# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# # RAG chain
# if vectorstore:
#     rag_chain = (
#         {"context": retriever | format_docs, "question": RunnablePassthrough()}
#         | prompt
#         | model
#         | StrOutputParser()
#     )

# def query_rag(question):
#     try:
#         if not vectorstore:
#             return "Vector database not available. Please ensure DSA.md file is present."
#         return rag_chain.invoke(question)
#     except Exception as e:
#         return f"Error from vector DB: {e}"

# def query_llm_sync(question, prepend_context=None):
#     try:
#         system_msg = (
#             f"You are an expert LeetCode assistant specializing in {language}. "
#             f"Provide clean, optimized solutions with detailed explanations. "
#             f"Include time and space complexity analysis. Use clear variable names and comments."
#         )

#         if prepend_context:
#             full_prompt = f"{system_msg}\n\nPrevious context:\n{prepend_context.strip()}\n\nFollow-up: {question}"
#         else:
#             full_prompt = f"{system_msg}\n\nQuestion: {question}"

#         response = model.invoke(full_prompt)
#         return response.content.strip() if hasattr(response, "content") else str(response).strip()
#     except Exception as e:
#         return f"LLM error: {e}"

# # Enhanced follow-up detection
# def is_followup_prompt(prompt: str) -> bool:
#     followup_phrases = [
#         r"\b(explain|elaborate|clarify|expand|more detail|why|how|what does.*mean)\b",
#         r"\b(previous|last|that code|that solution|earlier|continue|follow up|detail)\b",
#         r"\b(can you|could you|please|also|additionally)\b"
#     ]
#     return any(re.search(pattern, prompt.lower()) for pattern in followup_phrases)

# # Session state initialization
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []
# if "full_context" not in st.session_state:
#     st.session_state.full_context = ""

# # Display chat history with modern styling
# if st.session_state.chat_history:
#     st.markdown("""
#     <div class="chat-container">
#         <h3 style="color: #64ffda; margin-top: 0;">💬 Chat History</h3>
#     """, unsafe_allow_html=True)
    
#     for i, chat in enumerate(st.session_state.chat_history):
#         # User message
#         st.markdown(f"""
#         <div class="user-message">
#             <strong>🧑‍💻 You asked:</strong><br/>
#             {chat['question']}
#         </div>
#         """, unsafe_allow_html=True)
        
#         # Vector DB response
#         if chat["vector_db"] != "_(Follow-up — skipped vector DB)_":
#             st.markdown(f"""
#             <div class="assistant-message vector-response">
#                 <div class="vector-badge response-badge">📚 Knowledge Base</div>
#                 {chat["vector_db"]}
#             </div>
#             """, unsafe_allow_html=True)
        
#         # LLM response
#         response_type = "followup-badge" if "Follow-up" in chat.get("vector_db", "") else "llm-badge"
#         badge_text = "🔄 Follow-up" if "Follow-up" in chat.get("vector_db", "") else f"🤖 AI Assistant ({language})"
        
#         st.markdown(f"""
#         <div class="assistant-message llm-response">
#             <div class="{response_type} response-badge">{badge_text}</div>
#             {chat["llm"]}
#         </div>
#         """, unsafe_allow_html=True)
        
#         if i < len(st.session_state.chat_history) - 1:
#             st.markdown('<hr style="border: none; height: 1px; background: linear-gradient(90deg, transparent, #667eea, transparent); margin: 2rem 0;">', unsafe_allow_html=True)
    
#     st.markdown("</div>", unsafe_allow_html=True)

# # Enhanced input area
# st.markdown("""
# <div class="input-container">
#     <h4 style="color: #64ffda; margin-top: 0;">✨ Ask anything about LeetCode or DSA</h4>
# </div>
# """, unsafe_allow_html=True)

# with st.form(key="input_form", clear_on_submit=True):
#     user_input = st.text_area(
#         "",
#         placeholder="Type your question here... (e.g., 'How do I solve the Two Sum problem?' or 'Explain binary search')",
#         height=120,
#         max_chars=2000,
#         key="input_box",
#         help="💡 Tip: You can ask follow-up questions for more details!"
#     )
    
#     col1, col2, col3 = st.columns([1, 1, 1])
#     with col2:
#         submitted = st.form_submit_button("🚀 Get Answer", use_container_width=True)

# # Handle input with enhanced feedback
# if submitted and user_input.strip():
#     user_input_clean = user_input.strip()
#     is_followup = is_followup_prompt(user_input_clean) and len(st.session_state.chat_history) > 0

#     if is_followup:
#         # Show processing status
#         with st.spinner("🧠 Processing follow-up question..."):
#             progress_bar = st.progress(0)
#             for i in range(100):
#                 time.sleep(0.01)
#                 progress_bar.progress(i + 1)
            
#             llm_answer = query_llm_sync(user_input_clean, prepend_context=st.session_state.full_context)
#             progress_bar.empty()

#         st.session_state.chat_history.append({
#             "question": user_input_clean,
#             "vector_db": "_(Follow-up — skipped vector DB)_",
#             "llm": llm_answer
#         })

#     else:
#         col1, col2 = st.columns(2)
        
#         with col1:
#             with st.spinner("🔍 Searching knowledge base..."):
#                 progress1 = st.progress(0)
#                 for i in range(50):
#                     time.sleep(0.02)
#                     progress1.progress((i + 1) * 2)
#                 vector_db_answer = query_rag(user_input_clean)
#                 progress1.empty()

#         with col2:
#             with st.spinner("🤖 Generating AI response..."):
#                 progress2 = st.progress(0)
#                 for i in range(50):
#                     time.sleep(0.02)
#                     progress2.progress((i + 1) * 2)
#                 llm_answer = query_llm_sync(user_input_clean)
#                 progress2.empty()

#         # Add to context memory
#         st.session_state.full_context += f"Q: {user_input_clean}\nA: {llm_answer[:200]}...\n\n"

#         st.session_state.chat_history.append({
#             "question": user_input_clean,
#             "vector_db": vector_db_answer,
#             "llm": llm_answer
#         })

#     st.rerun()

# # Add some helpful suggestions when chat is empty
# if not st.session_state.chat_history:
#     st.markdown("""
#     <div class="chat-container hover-card">
#         <h4 style="color: #64ffda;">🌟 Try asking about:</h4>
#         <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin-top: 1rem;">
#             <div style="background: var(--accent-bg); padding: 1rem; border-radius: 10px; border: 1px solid var(--border-color);">
#                 <strong style="color: #4facfe;">🔍 Algorithm Problems</strong><br/>
#                 <span style="color: #a8b2d1;">Two Sum, Binary Search, DFS/BFS</span>
#             </div>
#             <div style="background: var(--accent-bg); padding: 1rem; border-radius: 10px; border: 1px solid var(--border-color);">
#                 <strong style="color: #64ffda;">📊 Data Structures</strong><br/>
#                 <span style="color: #a8b2d1;">Arrays, Trees, Graphs, Hash Maps</span>
#             </div>
#             <div style="background: var(--accent-bg); padding: 1rem; border-radius: 10px; border: 1px solid var(--border-color);">
#                 <strong style="color: #f5576c;">⚡ Time Complexity</strong><br/>
#                 <span style="color: #a8b2d1;">Big O analysis and optimization</span>
#             </div>
#         </div>
#     </div>
#     """, unsafe_allow_html=True)

# # Footer
# st.markdown("""
# <div style="text-align: center; margin-top: 3rem; padding: 2rem; color: #a8b2d1; border-top: 1px solid var(--border-color);">
#     <p>Built with ❤️ using Streamlit • Powered by LangChain & GROQ</p>
#     <p style="font-size: 0.9rem;">🚀 Master algorithms, ace your interviews</p>
# </div>
# """, unsafe_allow_html=True)



import os
import re
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import time
import json
from datetime import datetime


# Page Configuration
st.set_page_config(
    page_title="LeetCode AI Assistant", 
    page_icon="🚀", 
    layout="wide",
    initial_sidebar_state="expanded"
)


# Enhanced CSS for beautiful UI
st.markdown("""
<style>
/* Import fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* Root variables with enhanced colors */
:root {
    --primary-bg: #0a0b14;
    --secondary-bg: #161b2e;
    --accent-bg: #242942;
    --card-bg: #1e2139;
    --gradient-primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --gradient-accent: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
    --gradient-success: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
    --gradient-warning: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
    --text-primary: #ffffff;
    --text-secondary: #a8b2d1;
    --text-accent: #64ffda;
    --text-muted: #6c7293;
    --border-color: rgba(255, 255, 255, 0.1);
    --border-accent: rgba(100, 255, 218, 0.3);
    --shadow-light: 0 4px 16px rgba(0, 0, 0, 0.1);
    --shadow-medium: 0 8px 32px rgba(31, 38, 135, 0.37);
    --shadow-heavy: 0 20px 60px rgba(0, 0, 0, 0.4);
    --glow: 0 0 30px rgba(100, 255, 218, 0.4);
}

/* Global styles */
.stApp {
    background: var(--primary-bg);
    font-family: 'Inter', sans-serif;
    color: var(--text-primary);
}

/* Hide Streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display: none;}

/* Enhanced sidebar */
.css-1d391kg, .css-1cypcdb {
    background: linear-gradient(180deg, var(--secondary-bg) 0%, var(--accent-bg) 100%);
    border-right: 1px solid var(--border-color);
}

/* Main title with enhanced animation */
.main-title {
    background: var(--gradient-primary);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 3.5rem;
    font-weight: 800;
    text-align: center;
    margin: 1.5rem 0 1rem 0;
    letter-spacing: -0.02em;
    position: relative;
    animation: titleGlow 3s ease-in-out infinite alternate;
}

.subtitle {
    text-align: center;
    color: var(--text-secondary);
    font-size: 1.2rem;
    font-weight: 400;
    margin-bottom: 2rem;
    opacity: 0.9;
}

@keyframes titleGlow {
    from { 
        filter: brightness(1) drop-shadow(0 0 20px rgba(100, 255, 218, 0.3)); 
    }
    to { 
        filter: brightness(1.3) drop-shadow(0 0 30px rgba(100, 255, 218, 0.6)); 
    }
}

/* Enhanced chat container */
.chat-container {
    background: var(--card-bg);
    border-radius: 24px;
    padding: 2rem;
    margin: 1.5rem 0;
    border: 1px solid var(--border-color);
    backdrop-filter: blur(20px);
    box-shadow: var(--shadow-medium);
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
}

.chat-container::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: var(--gradient-primary);
    border-radius: 24px 24px 0 0;
}

.chat-container:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-heavy);
    border-color: var(--border-accent);
}

/* Enhanced message bubbles */
.message-container {
    margin: 1.5rem 0;
    animation: fadeInUp 0.4s ease-out;
}

.user-message {
    background: var(--gradient-primary);
    color: white;
    padding: 1.2rem 1.8rem;
    border-radius: 24px 24px 8px 24px;
    max-width: 85%;
    margin-left: auto;
    box-shadow: var(--shadow-light);
    font-weight: 500;
    line-height: 1.5;
    position: relative;
    overflow: hidden;
}

.user-message::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(45deg, rgba(255,255,255,0.1) 0%, transparent 100%);
    pointer-events: none;
}

.assistant-message {
    background: var(--card-bg);
    color: var(--text-primary);
    padding: 1.8rem;
    border-radius: 24px 24px 24px 8px;
    max-width: 90%;
    border: 1px solid var(--border-color);
    box-shadow: var(--shadow-light);
    position: relative;
    line-height: 1.6;
    border-left: 4px solid var(--text-accent);
}

@keyframes fadeInUp {
    from { 
        transform: translateY(30px); 
        opacity: 0; 
    }
    to { 
        transform: translateY(0); 
        opacity: 1; 
    }
}

/* Enhanced input area */
.input-container {
    background: var(--card-bg);
    border-radius: 20px;
    padding: 2rem;
    border: 1px solid var(--border-color);
    backdrop-filter: blur(20px);
    margin: 2rem 0;
    box-shadow: var(--shadow-light);
    transition: all 0.3s ease;
}

.input-container:hover {
    border-color: var(--border-accent);
    box-shadow: var(--shadow-medium);
}

/* Enhanced buttons */
.stButton button {
    background: var(--gradient-primary);
    color: white;
    border: none;
    border-radius: 30px;
    padding: 1rem 2.5rem;
    font-weight: 600;
    font-size: 1rem;
    transition: all 0.3s ease;
    box-shadow: var(--shadow-light);
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

.stButton button:hover {
    transform: translateY(-3px);
    box-shadow: 0 15px 45px rgba(102, 126, 234, 0.4);
    background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
}

/* Enhanced form elements */
.stSelectbox > div > div, .stTextArea textarea {
    background: var(--accent-bg) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.3s ease !important;
}

.stTextArea textarea:focus {
    border-color: var(--text-accent) !important;
    box-shadow: 0 0 0 2px rgba(100, 255, 218, 0.2) !important;
}

/* Enhanced metrics */
.metric-card {
    background: var(--card-bg);
    padding: 1.5rem;
    border-radius: 16px;
    border: 1px solid var(--border-color);
    text-align: center;
    margin: 0.5rem;
    transition: all 0.3s ease;
    box-shadow: var(--shadow-light);
}

.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-medium);
    border-color: var(--border-accent);
}

/* Status indicators */
.status-indicator {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-right: 10px;
    animation: statusPulse 2s infinite;
    box-shadow: 0 0 10px currentColor;
}

.status-online { background: #00ff88; color: #00ff88; }
.status-processing { background: #ffa500; color: #ffa500; }

@keyframes statusPulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.7; transform: scale(1.1); }
}

/* Code blocks */
.stCode, pre {
    background: var(--primary-bg) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    font-family: 'JetBrains Mono', monospace !important;
    box-shadow: var(--shadow-light) !important;
}

/* Enhanced badges */
.language-badge {
    background: var(--gradient-accent);
    color: white;
    padding: 0.4rem 1rem;
    border-radius: 25px;
    font-size: 0.85rem;
    font-weight: 600;
    display: inline-block;
    margin: 0.25rem 0;
    box-shadow: var(--shadow-light);
    letter-spacing: 0.5px;
}

/* Loading animation */
.loading-container {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 2rem;
    color: var(--text-accent);
    font-weight: 500;
}

.loading-dots {
    display: inline-block;
    margin-left: 0.5rem;
}

.loading-dots::after {
    content: '';
    animation: loadingDots 1.5s infinite;
}

@keyframes loadingDots {
    0%, 20% { content: '●'; }
    40% { content: '●●'; }
    60%, 100% { content: '●●●'; }
}

/* Suggestion cards */
.suggestion-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1.5rem;
    margin-top: 1.5rem;
}

.suggestion-card {
    background: var(--card-bg);
    padding: 1.5rem;
    border-radius: 16px;
    border: 1px solid var(--border-color);
    transition: all 0.3s ease;
    cursor: pointer;
    box-shadow: var(--shadow-light);
    position: relative;
    overflow: hidden;
}

.suggestion-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: var(--gradient-primary);
    transform: scaleX(0);
    transition: transform 0.3s ease;
}

.suggestion-card:hover {
    transform: translateY(-5px);
    box-shadow: var(--shadow-heavy);
    border-color: var(--border-accent);
}

.suggestion-card:hover::before {
    transform: scaleX(1);
}

/* Custom scrollbar */
::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-track {
    background: var(--primary-bg);
    border-radius: 5px;
}

::-webkit-scrollbar-thumb {
    background: var(--gradient-primary);
    border-radius: 5px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
}

/* Timestamp */
.timestamp {
    color: var(--text-muted);
    font-size: 0.75rem;
    margin-top: 0.5rem;
    font-weight: 400;
}

/* Progress bars */
.stProgress .st-bo {
    background: var(--gradient-primary);
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 4rem;
    padding: 2.5rem;
    color: var(--text-secondary);
    border-top: 1px solid var(--border-color);
    background: var(--secondary-bg);
    border-radius: 20px 20px 0 0;
}
</style>
""", unsafe_allow_html=True)


# Enhanced sidebar with modern design
st.sidebar.markdown("""
<div style="text-align: center; padding: 1.5rem;">
    <h2 style="color: #64ffda; margin: 0; font-weight: 600;">⚙️ Control Panel</h2>
    <div class="status-indicator status-online"></div>
    <span style="color: #a8b2d1; font-size: 0.9rem; font-weight: 500;">AI Assistant Online</span>
</div>
""", unsafe_allow_html=True)


# Language selection with enhanced styling
language = st.sidebar.selectbox(
    "🔤 Programming Language:",
    ["Python", "C++", "Java", "JavaScript", "Go"],
    index=0,
    help="Choose your preferred language for code solutions"
)

st.sidebar.markdown(f'<div class="language-badge">{language}</div>', unsafe_allow_html=True)

# Session metrics
st.sidebar.markdown("### 📊 Session Analytics")
col1, col2 = st.sidebar.columns(2)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div style="color: #4facfe; font-size: 1.8rem; font-weight: bold;">
            {len(st.session_state.get('chat_history', []))}
        </div>
        <div style="color: #a8b2d1; font-size: 0.8rem; font-weight: 500;">QUESTIONS</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div style="color: #64ffda; font-size: 1.2rem; font-weight: bold;">
            {language}
        </div>
        <div style="color: #a8b2d1; font-size: 0.8rem; font-weight: 500;">LANGUAGE</div>
    </div>
    """, unsafe_allow_html=True)

# Clear chat button
if st.sidebar.button("🗑️ Clear Chat History", use_container_width=True):
    st.session_state.chat_history = []
    st.session_state.memory = []
    st.rerun()

# Main title with enhanced design
st.markdown('<h1 class="main-title">🚀 LeetCode AI Assistant</h1>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Your intelligent companion for mastering Data Structures & Algorithms</div>', unsafe_allow_html=True)


# Load and split documents
@st.cache_resource
def load_docs():
    try:
        loader = TextLoader("DSA.md", encoding="utf-8")
        documents = loader.load()
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        return splitter.split_documents(documents)
    except Exception as e:
        st.error(f"📄 Could not load DSA.md file: {e}")
        return []

text = load_docs()

# Embeddings
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False}
    )

try:
    embeddings = get_embeddings()
    st.sidebar.success("✅ Knowledge Base Loaded")
except Exception as e:
    st.error(f"🔴 Embedding load error: {e}")
    st.stop()

# Vector store
@st.cache_resource
def get_vectorstore():
    if text:
        return FAISS.from_documents(text, embeddings)
    return None

vectorstore = get_vectorstore()
if vectorstore:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    st.sidebar.success("✅ Vector Database Ready")
else:
    st.sidebar.warning("⚠️ Vector DB not available")

# Load GROQ API Key
load_dotenv()
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
    st.sidebar.success("✅ API Connection Active")
except Exception:
    st.sidebar.error("🔴 GROQ_API_KEY not found!")
    groq_api_key = st.sidebar.text_input("Enter GROQ API Key:", type="password")
    if not groq_api_key:
        st.stop()

# Language model
model = ChatGroq(
    groq_api_key=groq_api_key,
    model="Llama3-8b-8192",
    temperature=0.1
)

# Enhanced prompt template
system_prompt = f"""You are an expert LeetCode assistant specializing in {language}. 

Your goal is to provide single, accurate, and comprehensive answers. Follow these guidelines:
1. Always respond in {language} unless specified otherwise
2. Provide clean, optimized solutions with clear explanations
3. Include time and space complexity analysis
4. Use descriptive variable names and comments
5. Give practical examples when helpful
6. Be concise but thorough

If the user asks about previous conversations or questions, refer to the chat memory provided in the context.
"""

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "memory" not in st.session_state:
    st.session_state.memory = []

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def create_memory_context():
    """Create a context string from chat memory for reference"""
    if not st.session_state.memory:
        return ""
    
    memory_context = "Previous conversation context:\n"
    for i, item in enumerate(st.session_state.memory[-10:], 1):  # Last 10 interactions
        memory_context += f"{i}. Q: {item['question'][:100]}...\n   A: {item['answer'][:200]}...\n"
    return memory_context

def get_smart_answer(question):
    """Get single intelligent answer combining vector DB and LLM"""
    try:
        # Get memory context
        memory_context = create_memory_context()
        
        # Check if question is about previous conversations
        is_memory_question = any(phrase in question.lower() for phrase in 
                                ['previous', 'earlier', 'first question', 'what did i ask', 'my question', 'before'])
        
        if is_memory_question and st.session_state.memory:
            # Handle memory-related questions
            full_prompt = f"""
            {system_prompt}
            
            {memory_context}
            
            Current question: {question}
            
            Please answer based on the previous conversation context provided above.
            """
            response = model.invoke(full_prompt)
            return response.content.strip()
        
        # For regular questions, use vector DB + LLM
        if vectorstore:
            # Get relevant documents
            docs = retriever.invoke(question)
            context = format_docs(docs)
            
            # Create comprehensive prompt
            full_prompt = f"""
            {system_prompt}
            
            Knowledge Base Context:
            {context}
            
            {memory_context}
            
            Question: {question}
            
            Provide a comprehensive answer using the knowledge base context and your expertise.
            """
        else:
            full_prompt = f"""
            {system_prompt}
            
            {memory_context}
            
            Question: {question}
            """
        
        response = model.invoke(full_prompt)
        return response.content.strip()
    
    except Exception as e:
        return f"I apologize, but I encountered an error while processing your question: {str(e)}"

# Display chat history
if st.session_state.chat_history:
    st.markdown("""
    <div class="chat-container">
        <h3 style="color: #64ffda; margin-top: 0; font-weight: 600;">💬 Conversation History</h3>
    """, unsafe_allow_html=True)
    
    for i, chat in enumerate(st.session_state.chat_history):
        st.markdown(f"""
        <div class="message-container">
            <div class="user-message">
                <strong>🧑‍💻 You:</strong><br/>
                {chat['question']}
                <div class="timestamp">{chat.get('timestamp', 'Just now')}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="message-container">
            <div class="assistant-message">
                <strong>🤖 AI Assistant ({language}):</strong><br/>
                {chat['answer']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if i < len(st.session_state.chat_history) - 1:
            st.markdown('<hr style="border: none; height: 1px; background: linear-gradient(90deg, transparent, #667eea, transparent); margin: 2rem 0;">', unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Enhanced input area
st.markdown("""
<div class="input-container">
    <h4 style="color: #64ffda; margin-top: 0; font-weight: 600;">✨ What would you like to know?</h4>
    <p style="color: #a8b2d1; margin-bottom: 1rem; font-size: 0.9rem;">Ask me anything about algorithms, data structures, or refer to our previous conversation!</p>
</div>
""", unsafe_allow_html=True)

with st.form(key="input_form", clear_on_submit=True):
    user_input = st.text_area(
        "",
        placeholder="Type your question here... (e.g., 'How do I solve Two Sum?' or 'What was my first question?')",
        height=120,
        max_chars=2000,
        key="input_box",
        help="💡 Pro tip: You can reference previous questions in our conversation!"
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        submitted = st.form_submit_button("🚀 Get Answer", use_container_width=True)

# Handle input
if submitted and user_input.strip():
    user_input_clean = user_input.strip()
    timestamp = datetime.now().strftime("%H:%M")
    
    # Show loading animation
    with st.empty():
        st.markdown("""
        <div class="loading-container">
            <span class="status-indicator status-processing"></span>
            Generating intelligent response<span class="loading-dots"></span>
        </div>
        """, unsafe_allow_html=True)
        
        # Get smart answer
        answer = get_smart_answer(user_input_clean)
        
        # Add to chat history and memory
        chat_entry = {
            "question": user_input_clean,
            "answer": answer,
            "timestamp": timestamp
        }
        
        st.session_state.chat_history.append(chat_entry)
        st.session_state.memory.append(chat_entry)
        
        # Clear loading and rerun
        st.empty()
        time.sleep(0.1)  # Small delay for smooth transition
    
    st.rerun()

# Suggestion cards when chat is empty
if not st.session_state.chat_history:
    st.markdown("""
    <div class="chat-container">
        <h4 style="color: #64ffda; margin-bottom: 1.5rem; font-weight: 600;">🌟 Popular Topics to Explore</h4>
        <div class="suggestion-grid">
            <div class="suggestion-card">
                <strong style="color: #4facfe; font-size: 1.1rem;">🔍 Algorithm Fundamentals</strong><br/>
                <span style="color: #a8b2d1;">Two Sum, Binary Search, Sorting Algorithms</span>
            </div>
            <div class="suggestion-card">
                <strong style="color: #64ffda; font-size: 1.1rem;">📊 Data Structures</strong><br/>
                <span style="color: #a8b2d1;">Arrays, LinkedLists, Trees, Graphs, Hash Maps</span>
            </div>
            <div class="suggestion-card">
                <strong style="color: #f5576c; font-size: 1.1rem;">⚡ Complexity Analysis</strong><br/>
                <span style="color: #a8b2d1;">Big O notation, Time & Space optimization</span>
            </div>
            <div class="suggestion-card">
                <strong style="color: #ffa500; font-size: 1.1rem;">🎯 Interview Prep</strong><br/>
                <span style="color: #a8b2d1;">Common patterns, Problem-solving strategies</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Enhanced footer
st.markdown("""
<div class="footer">
    <h4 style="color: #64ffda; margin-bottom: 1rem; font-weight: 600;">🚀 LeetCode AI Assistant</h4>
    <p style="margin-bottom: 0.5rem;">Built with ❤️ using Streamlit • Powered by LangChain & GROQ</p>
    <p style="font-size: 0.9rem; color: #6c7293;">Your intelligent companion for coding interview success</p>
</div>
""", unsafe_allow_html=True)