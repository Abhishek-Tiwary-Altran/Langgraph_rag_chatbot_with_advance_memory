from config import initialize_bedrock_client, initialize_embeddings, initialize_llm
from document_processor import create_vectorstore, create_retriever
from agents import (create_rag_chain, create_retrieval_grader, 
                    create_hallucination_grader, create_answer_grader)
from web_search import initialize_web_search_tool
from workflow import WorkflowManager
from bedrock_agentcore.memory import MemoryClient
from botocore.exceptions import ClientError
from datetime import datetime
import os
import logging

# os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"https://aps-workspaces.us-east-1.amazonaws.com/v1/metrics"

# Configure logging
logger = logging.getLogger(__name__)

def initialize_memory(username=None):
    """Initialize Bedrock AgentCore memory.
    
    Args:
        username: Optional username for user-specific session tracking
    
    Returns:
        Tuple of (memory_client, memory_id, actor_id, session_id)
    """
    try:
        client = MemoryClient(region_name="us-east-1")
        memory_name = "langgraph_rag"
        memory_id = None
        
        try:
            logger.info("Creating memory...")
            memory = client.create_memory_and_wait(
                name=memory_name,
                description="LangGraph RAG Assistant Memory",
                strategies=[],
                event_expiry_days=7,
                max_wait=300,
                poll_interval=10
            )
            memory_id = memory['id']
            logger.info(f"Memory created successfully with ID: {memory_id}")
        except ClientError as e:
            if e.response['Error']['Code'] == 'ValidationException' and "already exists" in str(e):
                memories = client.list_memories()
                memory_id = next((m['id'] for m in memories if m['id'].startswith(memory_name)), None)
                logger.info(f"Memory already exists. Using existing memory ID: {memory_id}")
            else:
                raise
        
        # Generate user-specific IDs
        if username:
            actor_id = f"user-{username}"
            session_id = f"session-{username}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        else:
            actor_id = f"user-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            session_id = f"rag-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        return client, memory_id, actor_id, session_id
    except Exception as e:
        logger.warning(f"Failed to initialize memory: {str(e)}")
        return None, None, None, None

def initialize_system(doc_splits=None, username=None):
    """Initialize all system components and return workflow manager.
    
    Args:
        doc_splits: Optional document splits to use for the vector store
                   If None, will use an empty vector store
        username: Optional username for user-specific session tracking
    """
    # Initialize AWS services
    bedrock_client = initialize_bedrock_client()
    embed_model = initialize_embeddings(bedrock_client)
    llm = initialize_llm(bedrock_client)
    
    # Initialize memory with username
    memory_client, memory_id, actor_id, session_id = initialize_memory(username)
    
    # Create vector store from documents if provided, otherwise create empty store
    if doc_splits:
        logger.info(f"Creating vector store with {len(doc_splits)} document chunks")
        vectorstore = create_vectorstore(doc_splits, embed_model)
    else:
        logger.info("Creating empty vector store")
        # Create empty vector store or load existing one if available
        persist_dir = "./chroma_db"
        if os.path.exists(persist_dir):
            from langchain_community.vectorstores import Chroma
            vectorstore = Chroma(
                persist_directory=persist_dir,
                embedding_function=embed_model,
                collection_name="user-documents"
            )
            logger.info("Loaded existing vector store")
        else:
            # Create an empty vector store
            vectorstore = create_vectorstore([], embed_model, collection_name="user-documents")
            logger.info("Created new empty vector store")
    
    retriever = create_retriever(vectorstore)

    # Initialize components
    rag_chain = create_rag_chain(llm, memory_client, memory_id, actor_id, session_id)
    retrieval_grader = create_retrieval_grader(llm)
    hallucination_grader = create_hallucination_grader(llm)
    answer_grader = create_answer_grader(llm)
    web_search_tool = initialize_web_search_tool()

    # Create and return workflow manager
    return WorkflowManager(
        retriever=retriever,
        rag_chain=rag_chain,
        retrieval_grader=retrieval_grader,
        hallucination_grader=hallucination_grader,
        answer_grader=answer_grader,
        web_search_tool=web_search_tool,
        vectorstore=vectorstore,
        memory_client=memory_client,
        memory_id=memory_id,
        actor_id=actor_id,
        session_id=session_id
    )

def main():
    """Main application entry point."""
    workflow_manager = initialize_system()
    
    # Run the application with a sample question
    from pprint import pprint
    inputs = {"question": "What is prompt engineering?"}
    
    try:
        for output in workflow_manager.create_workflow().stream(inputs):
            for key, value in output.items():
                pprint(f"Finished running: {key}")
        
        # Print the final generation
        if "generation" in value:
            print("\nFinal Answer:")
            pprint(value["generation"])
        else:
            print("No generation found in the output")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    
    inputs = {"question": "what was the last question and enhance the answer of last question"}
    
    try:
        for output in workflow_manager.create_workflow().stream(inputs):
            for key, value in output.items():
                pprint(f"Finished running: {key}")
        
        # Print the final generation
        if "generation" in value:
            print("\nFinal Answer:")
            pprint(value["generation"])
        else:
            print("No generation found in the output")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()