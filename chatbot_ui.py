import streamlit as st
from main import initialize_system
import time
import logging
import os
import json
from document_processor import process_file, add_documents_to_vectorstore
from config import initialize_bedrock_client, initialize_embeddings
from handlers.auth_handler import AuthHandler
from components.login_page import login_page
from utils.telemetry import set_session_context

# os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"https://aps-workspaces.us-east-1.amazonaws.com/v1/metrics"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chatbot_ui.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

os.environ["LANGSMITH_OTEL_ENABLED"] = "true"

def load_aws_credentials():
    """Load AWS credentials from config file."""
    try:
        config_path = os.path.join(os.path.dirname(__file__), 'config/aws_config.json')
        with open(config_path, 'r') as config_file:
            return json.load(config_file)
    except Exception as e:
        st.error(f"Error loading AWS credentials: {str(e)}")
        return {}

def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "embed_model" not in st.session_state:
        # Initialize embedding model for document processing
        bedrock_client = initialize_bedrock_client()
        st.session_state.embed_model = initialize_embeddings(bedrock_client)
    
    # Initialize workflow manager only after authentication
    if ("workflow_manager" not in st.session_state and 
        st.session_state.get('authenticated', False)):
        logger.info("Initializing workflow manager")
        username = st.session_state.get('username')
        st.session_state.workflow_manager = initialize_system(username=username)
        
        # Set session context for tracing
        workflow_manager = st.session_state.workflow_manager
        if hasattr(workflow_manager, 'session_id') and hasattr(workflow_manager, 'actor_id'):
            set_session_context(
                session_id=workflow_manager.session_id,
                user_id=username,
                actor_id=workflow_manager.actor_id
            )
            logger.info(f"Session context set for user: {username}")
        
        logger.info("Workflow manager initialized")
        # Load conversation history from memory if available
        load_memory_history()

def load_memory_history():
    """Load conversation history from memory."""
    workflow_manager = st.session_state.workflow_manager
    logger.info(f"Loading memory history. Memory client: {hasattr(workflow_manager, 'memory_client')}, Memory ID: {hasattr(workflow_manager, 'memory_id')}")
    
    if (hasattr(workflow_manager, 'memory_client') and workflow_manager.memory_client and 
        hasattr(workflow_manager, 'memory_id') and workflow_manager.memory_id and 
        not st.session_state.messages):  # Only load if no messages exist
        try:
            events = workflow_manager.memory_client.list_events(
                memory_id=workflow_manager.memory_id,
                actor_id=workflow_manager.actor_id,
                session_id=workflow_manager.session_id,
                max_results=20
            )
            
            # Convert memory events to chat messages
            for event in reversed(events):  # Reverse to get chronological order
                if 'messages' in event:
                    for message_pair in event['messages']:
                        if len(message_pair) >= 2:
                            content, role = message_pair[0], message_pair[1]
                            if role == "USER":
                                st.session_state.messages.append({"role": "user", "content": content})
                            elif role == "ASSISTANT":
                                st.session_state.messages.append({"role": "assistant", "content": content, "source": "Memory"})
            
            if events:
                logger.info(f"Loaded {len(events)} conversation events from memory")
        except Exception as e:
            logger.warning(f"Failed to load memory history: {str(e)}")
    else:
        logger.info("Memory not available or messages already exist")

def display_chat_history():
    """Display chat history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display source information if available
            if "source" in message and message["source"]:
                st.caption(f"Source: {message['source']}")

def process_uploaded_file(file):
    """Process an uploaded file and add it to the vector store.
    
    Args:
        file: Streamlit uploaded file object
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Process the file
        doc_splits = process_file(file)
        
        # Get workflow manager from session state
        workflow_manager = st.session_state.workflow_manager
        
        # Debug information
        logger.info(f"WorkflowManager attributes: {dir(workflow_manager)}")
        
        # Check if vectorstore exists in workflow manager
        if hasattr(workflow_manager, 'vectorstore') and workflow_manager.vectorstore is not None:
            logger.info("Found vectorstore in workflow manager")
            add_documents_to_vectorstore(workflow_manager.vectorstore, doc_splits)
            st.session_state.uploaded_files.append(file.name)
            return True
        else:
            # If no vectorstore in workflow manager, create one using the embedding model
            logger.warning("No vectorstore in workflow manager, creating a new one")
            from document_processor import create_vectorstore
            
            # Get embedding model from session state
            embed_model = st.session_state.embed_model
            
            # Create a new vectorstore with the document
            vectorstore = create_vectorstore(doc_splits, embed_model)
            
            # Update the workflow manager's vectorstore
            workflow_manager.vectorstore = vectorstore
            
            # Update the retriever to use the new vectorstore
            from document_processor import create_retriever
            workflow_manager.retriever = create_retriever(vectorstore)
            
            st.session_state.uploaded_files.append(file.name)
            return True
    except Exception as e:
        logger.error(f"Error processing file {file.name}: {str(e)}")
        logger.error(f"Exception details: {str(e)}", exc_info=True)
        return False

def process_user_input(user_input: str):
    """Process user input and generate response.
    
    This function displays the user's question immediately and then
    processes it to generate and display the assistant's response.
    
    Args:
        user_input: The user's question or input
    """
    logger.info(f"Processing user input: {user_input}")
    
    # Add user message to chat history and display it immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display the user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Get workflow manager from session state
    workflow_manager = st.session_state.workflow_manager
    
    # Create input for workflow
    inputs = {"question": user_input}
    
    # Process the question and display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        source_placeholder = st.empty()
        full_response = ""
        data_source = "Unknown"
        
        try:
            # Show "Thinking..." while processing
            message_placeholder.markdown("Thinking...")
            
            # Run the workflow
            final_output = None
            
            # Track processing steps and data source
            processing_steps = []
            last_node = None
            
            for output in workflow_manager.create_workflow().stream(inputs):
                for key, value in output.items():
                    last_node = key
                    processing_steps.append(f"Processing: {key}")
                    message_placeholder.markdown(f"Thinking...\n\n*{', '.join(processing_steps)}*")
                    
                    # Track data source based on the workflow path
                    if key == "websearch":
                        data_source = "Web Search"
                    elif key == "search_memory" or key == "grade_memory":
                        if data_source != "Web Search":  # Don't override web search
                            data_source = "Memory"
                    elif key == "retrieve" or key == "grade_documents":
                        if data_source != "Web Search":  # Don't override web search
                            data_source = "Knowledge Base"
                    
                    if isinstance(value, dict) and "generation" in value:
                        final_output = value["generation"]
            
            # Update source information based on the last processing step
            if final_output and isinstance(value, dict) and "documents" in value:
                docs = value["documents"]
                if isinstance(docs, str) and "Conversation History:" in docs:
                    data_source = "Memory"
                elif isinstance(docs, str) and ("Source:" in docs or "http" in docs):
                    data_source = "Web Search"
                elif last_node == "websearch":
                    data_source = "Web Search"
                elif docs and len(docs) > 0:
                    data_source = "Knowledge Base"
                else:
                    data_source = "Web Search"
            
            source_placeholder.caption(f"Source: {data_source}")
            
            if final_output:
                # Simulate typing effect
                for chunk in final_output.split():
                    full_response += chunk + " "
                    time.sleep(0.02)  # Slightly faster typing speed
                    message_placeholder.markdown(full_response)
            else:
                full_response = "I apologize, but I couldn't generate a response for that question."
                message_placeholder.markdown(full_response)
                
        except Exception as e:
            logger.error(f"Error processing input: {str(e)}")
            full_response = f"I apologize, but an error occurred: {str(e)}"
            message_placeholder.markdown(full_response)
            data_source = "Error"
    
    # Add assistant response to chat history with source information
    st.session_state.messages.append({
        "role": "assistant", 
        "content": full_response,
        "source": data_source
    })
    logger.info(f"Response generated from {data_source} and added to chat history")

def file_uploader_section():
    """Display file uploader section."""
    st.sidebar.header("Upload Documents")
    
    # File uploader
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF, DOCX, or TXT files", 
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        # Process button
        if st.sidebar.button("Process Files"):
            with st.sidebar.status("Processing files...") as status:
                success_count = 0
                for file in uploaded_files:
                    # Check if file already processed
                    if file.name in st.session_state.uploaded_files:
                        st.sidebar.info(f"File {file.name} already processed")
                        continue
                        
                    # Process file
                    st.sidebar.text(f"Processing {file.name}...")
                    if process_uploaded_file(file):
                        success_count += 1
                        st.sidebar.success(f"Successfully processed {file.name}")
                    else:
                        st.sidebar.error(f"Failed to process {file.name}")
                
                if success_count > 0:
                    status.update(label=f"Processed {success_count} files successfully!", state="complete")
                else:
                    status.update(label="No new files processed", state="error")
    
    # Display uploaded files
    if st.session_state.uploaded_files:
        st.sidebar.subheader("Uploaded Documents")
        for file in st.session_state.uploaded_files:
            st.sidebar.text(f"• {file}")

def main():
    st.set_page_config(
        page_title="AI Research Assistant",
        page_icon="🔍",
        layout="wide"
    )
    
    # Load AWS credentials and initialize auth handler
    credentials = load_aws_credentials()
    if not credentials:
        st.error("Failed to load AWS credentials. Please check config/aws_config.json")
        return
    
    auth_handler = AuthHandler(credentials)
    
    # Check authentication
    if not login_page(auth_handler):
        return
    
    st.title(f"AI Research Assistant - Welcome {st.session_state.username}!")
    
    # Add logout button
    if st.sidebar.button("Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    st.sidebar.title("About")
    st.sidebar.info(
        """
        This assistant combines RAG (Retrieval-Augmented Generation) with web search 
        capabilities and memory management to provide accurate and contextual answers.
        
        It can:
        - Answer questions from your uploaded documents
        - Search the web for additional information
        - Remember conversation history across sessions
        - Verify information to avoid hallucinations
        """
    )
    
    # Initialize session state
    init_session_state()
    
    # Check if workflow manager is initialized
    if "workflow_manager" not in st.session_state:
        st.info("Initializing system...")
        return
    
    # File uploader section
    file_uploader_section()
    
    # Add system information
    with st.sidebar.expander("System Information"):
        st.write("""
        - Uses LangGraph for workflow orchestration
        - Combines vector database retrieval with web search
        - Implements multiple quality checks for answers
        - Uses Bedrock AgentCore for conversation memory
        """)
        
        # Display memory status
        workflow_manager = st.session_state.get('workflow_manager')
        if workflow_manager and hasattr(workflow_manager, 'memory_id') and workflow_manager.memory_id:
            st.success(f"Memory Active: {workflow_manager.memory_id[:8]}...")
        else:
            st.warning("Memory: Not Available")
        
    # Add data sources information
    with st.sidebar.expander("Data Sources"):
        st.write("""
        - **Knowledge Base**: Information from your uploaded documents
        - **Web Search**: Real-time information from the web via search API
        - **Memory**: Previous conversation context and history
        """)
        
    # Add memory management controls
    with st.sidebar.expander("Memory Management"):
        workflow_manager = st.session_state.get('workflow_manager')
        
        # Debug information
        if workflow_manager:
            st.write(f"Memory client exists: {hasattr(workflow_manager, 'memory_client')}")
            if hasattr(workflow_manager, 'memory_client'):
                st.write(f"Memory client value: {workflow_manager.memory_client is not None}")
            if hasattr(workflow_manager, 'memory_id'):
                st.write(f"Memory ID: {workflow_manager.memory_id}")
        
        if workflow_manager and hasattr(workflow_manager, 'memory_client') and workflow_manager.memory_client:
            if st.button("Clear Memory"):
                try:
                    # Clear session messages
                    st.session_state.messages = []
                    st.success("Memory cleared successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to clear memory: {str(e)}")
        else:
            st.info("Memory management not available")

    # Display chat history
    display_chat_history()

    # Chat input
    if user_input := st.chat_input("What would you like to know?"):
        process_user_input(user_input)

if __name__ == "__main__":
    main()