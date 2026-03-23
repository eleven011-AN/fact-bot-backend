import os

def retrieve_evidence(claim: str) -> list:
    tavily_key = os.getenv("TAVILY_API_KEY")
    
    if tavily_key:
        from langchain_community.tools.tavily_search import TavilySearchResults
        tool = TavilySearchResults(max_results=3, tavily_api_key=tavily_key)
        try:
            return tool.invoke({"query": claim})
        except Exception as e:
            print(f"Tavily Error: {e}")
            return []
    else:
        # Fallback to duckduckgo
        from langchain_community.tools import DuckDuckGoSearchResults
        from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
        import json
        
        wrapper = DuckDuckGoSearchAPIWrapper(max_results=3)
        tool = DuckDuckGoSearchResults(api_wrapper=wrapper, output_format="list")
        try:
            results = tool.invoke(claim)
            return results
        except Exception as e:
            print(f"DDG Error: {e}")
            return []
