import streamlit as st
import asyncio
from rag_agent import run_parallel_agent
from utils import parse_query_string
import json
import nest_asyncio
import sys
import os

# Disable Streamlit's file watcher to prevent PyTorch conflicts
os.environ['STREAMLIT_SERVER_WATCH_DIRS'] = 'false'

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Set page config
st.set_page_config(
    page_title="Career Guidance Assistant",
    page_icon="🎓",
    layout="wide"
)

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'pending_user_message' not in st.session_state:
    st.session_state.pending_user_message = None

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-container {
        margin-bottom: 100px;
        padding-bottom: 20px;
    }
    .chat-message {
        display: inline-block;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
        border-radius: 12px;
        font-size: 1rem;
        line-height: 1.4;
        max-width: 60%;
        word-wrap: break-word;
    }
    .chat-message.assistant {
        background-color: #f0f2f6;
        color: #1f1f1f;
    }
    .chat-message.user {
        background-color: #2E4057;
        color: #ffffff;
    }
    .main-header {
        font-size: 2.5rem;
        color: #2E4057;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">Career Guidance Assistant</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("About")
    st.markdown("""
        This career guidance system combines specific insights from professionals with general career advice.
        Ask any questions about careers, industries, or professional development.
    """)
    
    # Optional filters
    st.markdown("### Filters")
    industry_filter = st.multiselect(
        "Filter by Industry",
        ["Technology", "Healthcare", "Finance", "Education", "Media", "Engineering"],
        default=[]
    )
    
    takeaways_filter = st.multiselect(
        "Filter by Key Takeaways",
        ["Skills", "Education", "Experience", "Networking", "Challenges", "Opportunities"],
        default=[]
    )
    
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.pending_user_message = None
        st.rerun()

# Chat interface
with st.container():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(
                f'<div style="display: flex; justify-content: flex-end;">'
                f'<div class="chat-message user">{message["content"]}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div style="display: flex; justify-content: flex-start;">'
                f'<div class="chat-message assistant">{message["content"]}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
    st.markdown('</div>', unsafe_allow_html=True)

# Chat input
user_input = st.chat_input("Ask about careers...")
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.pending_user_message = user_input
    st.rerun()

# Process the query when there's a pending message
if st.session_state.pending_user_message:
    # Create the query structure
    query_data = {
        "content_string_query": st.session_state.pending_user_message,
        "industry_filter": industry_filter,
        "takeaways_filter": takeaways_filter
    }
    
    # Convert to JSON string
    query_string = json.dumps(query_data)
    
    # Show loading state
    with st.spinner("Analyzing your question and gathering insights..."):
        try:
            # Create new event loop for this thread
            if sys.platform == 'win32':
                loop = asyncio.ProactorEventLoop()
            else:
                loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run the parallel agent with the query string
            result = loop.run_until_complete(run_parallel_agent(query_string=query_string))
            
            # Clean up
            loop.close()
            
            # Check if result is None or empty
            if result is None or result.strip() == "":
                result = "I apologize, but I couldn't generate a response at this time. Please try asking your question again."
            
            # Add the response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": result})
            st.session_state.pending_user_message = None
            st.rerun()
            
        except Exception as e:
            error_message = f"I apologize, but an error occurred while processing your request: {str(e)}"
            st.session_state.chat_history.append({"role": "assistant", "content": error_message})
            st.session_state.pending_user_message = None
            st.rerun()

# Add footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>This career guidance system combines specific insights from professionals with general career advice.</p>
        <p>Powered by RAG (Retrieval-Augmented Generation) technology</p>
    </div>
""", unsafe_allow_html=True) 