import os
from langchain_community.tools.tavily_search import TavilySearchResults


def initialize_web_search_tool():
    """Initialize web search tool."""
    os.environ['TAVILY_API_KEY'] = os.getenv("TAVILY_API_KEY")
    return TavilySearchResults(k=3)
