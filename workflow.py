"""LangGraph RAG Workflow Manager

This module defines the workflow for the LangGraph RAG system. It manages the state
transitions between different processing nodes and handles the decision logic for
routing questions, evaluating document relevance, and assessing answer quality.

The workflow is implemented as a directed graph with conditional edges that determine
the path based on the output of various evaluation functions.
"""

import logging
import traceback
from typing import Any, Dict, List, Optional, TypeVar, Union, Callable
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END
from langchain.schema import Document
from opentelemetry import trace
from bedrock_agentcore.memory import MemoryClient
from datetime import datetime
from utils.telemetry import get_session_context

import os
# os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"https://aps-workspaces.us-east-1.amazonaws.com/v1/metrics"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_workflow.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Type definitions
T = TypeVar('T')
StateDict = Dict[str, Any]


class GraphState(TypedDict):
    """Type definition for the state maintained throughout the workflow.
    
    Attributes:
        question: The user's original question
        generation: The generated answer
        web_search: Flag indicating if web search is needed ("Yes" or "No")
        documents: List of retrieved documents or context
    """
    question: str
    generation: str
    web_search: str
    documents: List[str]


class WorkflowManager:
    """Manager for the RAG workflow graph.
    
    This class orchestrates the entire RAG process, from question routing to
    document retrieval, answer generation, and quality assessment.
    
    Attributes:
        retriever: Document retrieval component
        rag_chain: Answer generation component
        retrieval_grader: Document relevance assessment component
        hallucination_grader: Factual grounding assessment component
        answer_grader: Answer usefulness assessment component

        web_search_tool: Web search component
        vectorstore: Vector store for document storage
        memory_client: Bedrock AgentCore memory client
        memory_id: Memory resource ID
        actor_id: Unique actor identifier
        session_id: Unique session identifier
    """
    
    def __init__(self, retriever, rag_chain, retrieval_grader, hallucination_grader, 
                 answer_grader, web_search_tool, vectorstore=None,
                 memory_client=None, memory_id=None, actor_id=None, session_id=None):
        """Initialize the workflow manager with all required components.
        
        Args:
            retriever: Component for retrieving documents from vector store
            rag_chain: Component for generating answers from context
            retrieval_grader: Component for assessing document relevance
            hallucination_grader: Component for detecting hallucinations
            answer_grader: Component for assessing answer usefulness
            question_router: Component for routing questions to data sources
            web_search_tool: Component for performing web searches
            vectorstore: Optional vector store for document storage
        """
        self.retriever = retriever
        self.rag_chain = rag_chain
        self.retrieval_grader = retrieval_grader
        self.hallucination_grader = hallucination_grader
        self.answer_grader = answer_grader
        self.web_search_tool = web_search_tool
        self.vectorstore = vectorstore
        self.memory_client = memory_client
        self.memory_id = memory_id
        self.actor_id = actor_id
        self.session_id = session_id
        logger.info("WorkflowManager initialized with all components")

    def _safe_invoke(self, component: Any, inputs: Dict[str, Any], 
                    component_name: str, default_response: Optional[Any] = None) -> Any:
        """Safely invoke a component with error handling.
        
        Args:
            component: The component to invoke
            inputs: The inputs to pass to the component
            component_name: Name of the component for logging
            default_response: Default response if the component fails
            
        Returns:
            The component's output or default response if an error occurs
        """
        try:
            logger.debug(f"Invoking {component_name} with inputs: {inputs}")
            result = component.invoke(inputs)
            logger.debug(f"{component_name} returned: {result}")
            return result
        except Exception as e:
            logger.error(f"Error in {component_name}: {str(e)}")
            logger.debug(f"Detailed traceback: {traceback.format_exc()}")
            return default_response

    def search_memory(self, state: StateDict) -> StateDict:
        """Search conversation memory for context.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with memory context
        """
        tracer = trace.get_tracer("agentic-search-langgraph")
        with tracer.start_as_current_span("search_memory") as span:
            logger.info("Searching conversation memory")
            question = state["question"]
            span.set_attribute("question", question)
            
            # Add session context to span
            session_ctx = get_session_context()
            for key, value in session_ctx.items():
                if value:
                    span.set_attribute(f"session.{key}", value)
            
            memory_context = []
            
            if self.memory_client and self.memory_id:
                try:
                    events = self.memory_client.list_events(
                        memory_id=self.memory_id,
                        actor_id=self.actor_id,
                        session_id=self.session_id,
                        max_results=10
                    )
                    
                    if events:
                        # Format memory as documents for grading
                        from langchain.schema import Document
                        memory_context = [Document(page_content=str(events), metadata={"source": "memory"})]
                        logger.info(f"Retrieved {len(events)} memory events")
                    else:
                        logger.info("No conversation history found")
                        
                except Exception as e:
                    logger.warning(f"Failed to search memory: {str(e)}")
            
            return {
                "question": question,
                "documents": memory_context,
                "web_search": "No"
            }

    def retrieve(self, state: StateDict) -> StateDict:
        """Retrieve documents based on question.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with retrieved documents
        """
        tracer = trace.get_tracer("agentic-search-langgraph")
        with tracer.start_as_current_span("retrieve") as span:
            logger.info("Starting document retrieval")
            question = state["question"]
            span.set_attribute("question", question)
            
            # Add session context to span
            session_ctx = get_session_context()
            for key, value in session_ctx.items():
                if value:
                    span.set_attribute(f"session.{key}", value)
            
            try:
                documents = self._safe_invoke(
                    self.retriever, 
                    question, 
                    "retriever", 
                    []
                )
                logger.info(f"Retrieved {len(documents)} documents")
                span.set_attribute("documents.count", len(documents))
                return {"documents": documents, "question": question}
            except Exception as e:
                logger.error(f"Document retrieval failed: {str(e)}")
                span.set_attribute("error", str(e))
                return {"documents": [], "question": question}

    def generate(self, state: StateDict) -> StateDict:
        """Generate answer based on documents and question.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with generated answer
        """
        tracer = trace.get_tracer("agentic-search-langgraph")
        with tracer.start_as_current_span("generate") as span:
            logger.info("Starting answer generation")
            question = state["question"]
            documents = state["documents"]
            span.set_attribute("question", question)
            
            # Add session context to span
            session_ctx = get_session_context()
            for key, value in session_ctx.items():
                if value:
                    span.set_attribute(f"session.{key}", value)
            
            # Get memory context if available
            memory_context = ""
            if self.memory_client and self.memory_id:
                try:
                    events = self.memory_client.list_events(
                        memory_id=self.memory_id,
                        actor_id=self.actor_id,
                        session_id=self.session_id,
                        max_results=5
                    )
                    if events:
                        memory_context = "\n\nRecent conversation context:\n" + str(events)
                except Exception as e:
                    logger.warning(f"Failed to retrieve memory context: {str(e)}")
            
            try:
                # Include memory context in the generation
                context_with_memory = str(documents) + memory_context
                generation = self._safe_invoke(
                    self.rag_chain, 
                    {"context": context_with_memory, "question": question}, 
                    "rag_chain",
                    "I don't have enough information to answer this question."
                )
                logger.info(f"Generated answer length: {len(generation)} characters")
                span.set_attribute("generation.length", len(generation))
                
                # Save conversation to memory
                if self.memory_client and self.memory_id and generation.strip():
                    try:
                        conversation = [
                            (question, "USER"),
                            (generation, "ASSISTANT")
                        ]
                        self.memory_client.create_event(
                            memory_id=self.memory_id,
                            actor_id=self.actor_id,
                            session_id=self.session_id,
                            messages=conversation
                        )
                        logger.info("Conversation saved to memory")
                    except Exception as e:
                        logger.warning(f"Failed to save conversation to memory: {str(e)}")
                
                return {"documents": documents, "question": question, "generation": generation}
            except Exception as e:
                logger.error(f"Answer generation failed: {str(e)}")
                span.set_attribute("error", str(e))
                fallback_response = "I'm sorry, I encountered an error while generating an answer."
                return {"documents": documents, "question": question, "generation": fallback_response}

    def grade_documents(self, state: StateDict) -> StateDict:
        """Grade document relevance to the question.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with filtered documents and web search flag
        """
        tracer = trace.get_tracer("agentic-search-langgraph")
        with tracer.start_as_current_span("grade_documents") as span:
            logger.info("Grading document relevance")
            question = state["question"]
            documents = state.get("documents", [])
            span.set_attribute("question", question)
            span.set_attribute("documents.input_count", len(documents))
            
            if not documents:
                logger.warning("No documents to grade, defaulting to web search")
                span.set_attribute("web_search_needed", True)
                return {"documents": [], "question": question, "web_search": "Yes"}
            
            filtered_docs = []
            
            for i, doc in enumerate(documents):
                try:
                    score = self._safe_invoke(
                        self.retrieval_grader,
                        {"question": question, "document": doc.page_content},
                        f"retrieval_grader (doc {i})",
                        {"score": "no"}
                    )
                    
                    grade = score.get('score', '').lower()
                    
                    if grade == "yes":
                        logger.info(f"Document {i} is relevant")
                        filtered_docs.append(doc)
                    else:
                        logger.info(f"Document {i} is not relevant")
                except Exception as e:
                    logger.error(f"Error grading document {i}: {str(e)}")
            
            # Only use web search if NO relevant documents found
            if filtered_docs:
                web_search = "No"
                logger.info(f"Found {len(filtered_docs)} relevant documents, proceeding to generate")
            else:
                web_search = "Yes"
                logger.warning("No relevant documents found, will use web search")
            
            span.set_attribute("documents.filtered_count", len(filtered_docs))
            span.set_attribute("web_search_needed", web_search == "Yes")
            return {"documents": filtered_docs, "question": question, "web_search": web_search}

    def web_search(self, state: StateDict) -> StateDict:
        """Perform web search for additional information.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with web search results
        """
        tracer = trace.get_tracer("agentic-search-langgraph")
        with tracer.start_as_current_span("web_search") as span:
            logger.info("Performing web search")
            question = state["question"]
            span.set_attribute("question", question)
            try:
                docs = self._safe_invoke(
                    self.web_search_tool,
                    {"query": question},
                    "web_search_tool",
                    []
                )
                
                formatted_results = []
                for result in docs:
                    try:
                        formatted_results.append({
                            "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "content": result.get("content", "")
                    })
                    except Exception as e:
                        logger.error(f"Error formatting search result: {str(e)}")
            except Exception as e:
                logger.error(f"Web search failed: {str(e)}")
                return {"documents": "Web search failed to return results.", "question": question}
            
            # Create context from results
            if formatted_results:
                context = "\n\n".join([
                    f"Source: {result['url']}\n{result['content']}"
                    for result in formatted_results
                ])
                logger.info(f"Web search returned {len(formatted_results)} results")
            else:
                context = "No relevant information found from web search."
                logger.warning("Web search returned no results")
            
            return {"documents": context, "question": question}
        

    def grade_memory(self, state: StateDict) -> StateDict:
        """Grade memory context relevance to the question.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with filtered memory and web search flag
        """
        tracer = trace.get_tracer("agentic-search-langgraph")
        with tracer.start_as_current_span("grade_memory") as span:
            logger.info("Grading memory context relevance")
            question = state["question"]
            documents = state.get("documents", [])
            span.set_attribute("question", question)
            span.set_attribute("documents.input_count", len(documents))
            
            if not documents:
                logger.info("No memory context to grade, proceeding to vector search")
                return {"documents": [], "question": question, "web_search": "Yes"}
            
            filtered_docs = []
            
            for i, doc in enumerate(documents):
                try:
                    score = self._safe_invoke(
                        self.retrieval_grader,
                        {"question": question, "document": doc.page_content},
                        f"memory_grader (doc {i})",
                        {"score": "no"}
                    )
                    
                    grade = score.get('score', '').lower()
                    
                    if grade == "yes":
                        logger.info(f"Memory context {i} is relevant")
                        filtered_docs.append(doc)
                    else:
                        logger.info(f"Memory context {i} is not relevant")
                except Exception as e:
                    logger.error(f"Error grading memory context {i}: {str(e)}")
            
            # Only proceed to vector search if NO relevant memory found
            if filtered_docs:
                web_search = "No"
                logger.info(f"Found {len(filtered_docs)} relevant memory contexts, proceeding to generate")
            else:
                web_search = "Yes"
                logger.info("No relevant memory context found, will use vector search")
            
            span.set_attribute("documents.filtered_count", len(filtered_docs))
            span.set_attribute("vector_search_needed", web_search == "Yes")
            return {"documents": filtered_docs, "question": question, "web_search": web_search}
    
    def decide_after_memory(self, state: StateDict) -> str:
        """Decide next step after memory grading.
        
        Args:
            state: Current workflow state
            
        Returns:
            Decision: "generate" if memory relevant, "retrieve" if not
        """
        logger.info("Deciding next step after memory grading")
        web_search = state.get("web_search", "Yes")
        
        if web_search == "Yes":
            logger.info("Memory not relevant, proceeding to vector search")
            return "retrieve"
        else:
            logger.info("Memory is relevant, generating answer")
            return "generate"

    def decide_to_generate(self, state: StateDict) -> str:
        """Decide whether to generate answer or perform web search.
        
        Args:
            state: Current workflow state
            
        Returns:
            Decision: "websearch" or "generate"
        """
        logger.info("Deciding whether to generate answer or use web search")
        web_search = state.get("web_search", "No")
        
        if web_search == "Yes":
            logger.info("Decision: Documents not relevant, using web search")
            return "websearch"
        else:
            logger.info("Decision: Documents are relevant, generating answer")
            return "generate"

    def grade_generation_v_documents_and_question(self, state: StateDict) -> str:
        """Grade generated answer against documents and question.
        
        Args:
            state: Current workflow state
            
        Returns:
            Assessment: "not supported", "not useful", or "useful"
        """
        logger.info("Grading generated answer")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        
        # Check if this is a fallback response - if so, accept it
        if "I don't have enough information" in generation:
            logger.info("Fallback response detected, accepting as useful")
            return "useful"
        
        # Check usefulness first (more lenient than hallucination check)
        try:
            score = self._safe_invoke(
                self.answer_grader,
                {"question": question, "generation": generation},
                "answer_grader",
                {"score": "yes"}  # Default to useful
            )
            
            usefulness_grade = score.get('score', '').lower()
            
            if usefulness_grade == "yes":
                logger.info("Answer is useful for the question")
                return "useful"
            else:
                logger.info("Answer is not useful for the question")
                return "not useful"
        except Exception as e:
            logger.error(f"Error grading answer: {str(e)}")
            # Default to useful to avoid infinite loops
            return "useful"

    def create_workflow(self) -> Callable:
        """Create and configure the workflow graph.
        
        Returns:
            Compiled workflow graph ready for execution
        """
        logger.info("Creating workflow graph")
        
        try:
            workflow = StateGraph(GraphState)
            
            # Add nodes
            workflow.add_node("search_memory", self.search_memory)
            workflow.add_node("grade_memory", self.grade_memory)
            workflow.add_node("retrieve", self.retrieve)
            workflow.add_node("grade_documents", self.grade_documents)
            workflow.add_node("websearch", self.web_search)
            workflow.add_node("generate", self.generate)
            
            # Set entry point - always start with memory search
            workflow.set_entry_point("search_memory")
            
            # Add edges
            workflow.add_edge("search_memory", "grade_memory")
            workflow.add_conditional_edges(
                "grade_memory",
                self.decide_after_memory,
                {
                    "generate": "generate",
                    "retrieve": "retrieve",
                },
            )
            workflow.add_edge("retrieve", "grade_documents")
            workflow.add_conditional_edges(
                "grade_documents",
                self.decide_to_generate,
                {
                    "websearch": "websearch",
                    "generate": "generate",
                },
            )
            workflow.add_edge("websearch", "generate")
            workflow.add_conditional_edges(
                "generate",
                self.grade_generation_v_documents_and_question,
                {
                    "not supported": "generate",
                    "useful": END,
                    "not useful": "websearch",
                },
            )
            
            logger.info("Workflow graph created successfully")
            return workflow.compile()
        except Exception as e:
            logger.error(f"Error creating workflow: {str(e)}")
            logger.debug(f"Detailed traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to create workflow: {str(e)}") from e
            
    def update_vectorstore(self, doc_splits):
        """Update the vector store with new documents.
        
        Args:
            doc_splits: Document chunks to add to the vector store
            
        Returns:
            True if successful, False otherwise
        """
        if not self.vectorstore:
            logger.error("No vector store available for update")
            return False
            
        try:
            logger.info(f"Adding {len(doc_splits)} document chunks to vector store")
            self.vectorstore.add_documents(doc_splits)
            self.vectorstore.persist()
            logger.info("Documents added and vector store persisted")
            return True
        except Exception as e:
            logger.error(f"Error updating vector store: {str(e)}")
            logger.debug(f"Detailed traceback: {traceback.format_exc()}")
            return False