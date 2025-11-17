"""Agent Definitions for LangGraph RAG System

This module contains the agent definitions for the LangGraph RAG system.
It defines various specialized agents that handle different aspects of the
retrieval-augmented generation process, including question routing, document
retrieval, answer generation, and various quality assessment functions.

Each agent is implemented as a chain of components: a prompt template,
a language model, and an output parser.
"""

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from opentelemetry import trace
from utils.telemetry import get_session_context

def create_rag_chain(llm, memory_client=None, memory_id=None, actor_id=None, session_id=None):
    """Create a retrieval-augmented generation (RAG) chain with memory support.
    
    This agent generates answers to questions based on provided context and
    can access conversation history for better contextual responses.
    
    Args:
        llm: The language model to use for answer generation
        memory_client: Optional memory client for conversation history
        memory_id: Memory resource ID
        actor_id: Actor identifier
        session_id: Session identifier
        
    Returns:
        A chain that takes a question and context and returns a generated answer
    """
    tracer = trace.get_tracer("agentic-search-langgraph")
    
    # Create memory retrieval tool if memory is available
    if memory_client and memory_id:
        @tool
        def get_conversation_history():
            """Retrieve recent conversation history for context"""
            try:
                events = memory_client.list_events(
                    memory_id=memory_id,
                    actor_id=actor_id,
                    session_id=session_id,
                    max_results=5
                )
                return events
            except Exception as e:
                return f"Error retrieving history: {str(e)}"
        
        # Bind tool to LLM
        llm_with_tools = llm.bind_tools([get_conversation_history])
    else:
        llm_with_tools = llm
    
    def traced_rag(inputs):
        with tracer.start_as_current_span("rag_generation") as span:
            question = inputs.get("question", "")
            context = inputs.get("context", "")
            span.set_attribute("question", question)
            span.set_attribute("context.length", len(str(context)))
            
            # Add session context to span
            session_ctx = get_session_context()
            for key, value in session_ctx.items():
                if value:
                    span.set_attribute(f"session.{key}", value)
            
            # Enhanced system message for memory-aware responses
            system_message = """You are an AI research assistant for question-answering tasks.
            Use the retrieved context to answer questions accurately and concisely.
            If you have access to conversation history tools, use them to provide contextual responses
            that reference previous questions or build upon earlier discussions.
            If you don't know the answer, say so. Keep responses to three sentences maximum."""
            
            # Create messages for the LLM
            messages = [
                SystemMessage(content=system_message),
                HumanMessage(content=f"Question: {question}\nContext: {context}")
            ]
            
            result = llm_with_tools.invoke(messages)
            
            # Debug logging
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"LLM result type: {type(result)}")
            logger.info(f"LLM result attributes: {dir(result)}")
            if hasattr(result, 'content'):
                logger.info(f"LLM result.content: '{result.content}'")
            
            # Extract content from the response
            if hasattr(result, 'content'):
                response_content = result.content
            elif hasattr(result, 'text'):
                response_content = result.text
            elif isinstance(result, str):
                response_content = result
            else:
                response_content = str(result)
            
            # Ensure we have a non-empty response
            if not response_content or not response_content.strip():
                logger.warning("Empty response from LLM, using fallback")
                response_content = "I don't have enough information to answer this question."
            
            logger.info(f"Final response content: '{response_content[:100]}...'")
            span.set_attribute("generation.length", len(response_content))
            return response_content
    
    class TracedChain:
        def invoke(self, inputs):
            return traced_rag(inputs)
    
    return TracedChain()


def create_retrieval_grader(llm):
    """Create a document relevance grading agent.
    
    This agent assesses whether a retrieved document is relevant to the
    user's question. It uses a lenient approach to avoid filtering out
    potentially useful documents.
    
    Args:
        llm: The language model to use for relevance assessment
        
    Returns:
        A chain that evaluates document relevance and returns a JSON with
        a 'score' key containing either 'yes' or 'no'
    """
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance 
        of a retrieved document to a user question. If the document contains keywords related to the user question, 
        grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explaination.
         <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the retrieved document: \n\n {document} \n\n
        Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question", "document"],
    )
    return prompt | llm | JsonOutputParser()


def create_hallucination_grader(llm):
    """Create a hallucination detection agent.
    
    This agent checks whether a generated answer is factually grounded in
    the provided documents. It helps prevent the system from generating
    information not supported by the retrieved context.
    
    Args:
        llm: The language model to use for hallucination detection
        
    Returns:
        A chain that evaluates factual grounding and returns a JSON with
        a 'score' key containing either 'yes' or 'no'
    """
    prompt = PromptTemplate(
        template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether 
        an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate 
        whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a 
        single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here are the facts:
        \n ------- \n
        {documents} 
        \n ------- \n
        Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["generation", "documents"],
    )
    return prompt | llm | JsonOutputParser()


def create_answer_grader(llm):
    """Create an answer usefulness assessment agent.
    
    This agent evaluates whether a generated answer is useful for resolving
    the user's question, regardless of factual accuracy. It ensures the
    system's responses are relevant to what the user asked.
    
    Args:
        llm: The language model to use for usefulness assessment
        
    Returns:
        A chain that evaluates answer usefulness and returns a JSON with
        a 'score' key containing either 'yes' or 'no'
    """
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an 
        answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is 
        useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
         <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
        \n ------- \n
        {generation} 
        \n ------- \n
        Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["generation", "question"],
    )
    return prompt | llm | JsonOutputParser()