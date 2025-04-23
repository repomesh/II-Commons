from fastmcp import FastMCP
import requests
import os

mcp = FastMCP("Common Ground MCP Server", dependencies=["requests"], request_timeout=300)

API_SERVER_BASE_URL = os.environ.get("API_SERVER_BASE_URL", "http://localhost:8080")


def search(query: str, max_results: int = 20, rerank: bool = False, refine_query: bool = False)->str:
    try:
        headers = {"Content-Type": "application/json"}
        data = {
            "query": query,
            "max_results": max_results,
            "options": {
                "rerank": rerank,
                "refine_query": refine_query
            },
        }
        response = requests.post(f"{API_SERVER_BASE_URL}/search", json=data, headers=headers)
        if response.status_code == 200:
            resp = response.json()
            results = resp.get("results", [])
            images = resp.get("images", [])
            
            ret = ""
            for res in results:
                ret += f"Title: {res['title']}\n"
                ret += f"Content: {res['text']}\n"
                ret += f"URL: {res['url']}\n\n"
            for img in images:
                ret += f"Image URL: {img['url']}\n"
                ret += f"Image Caption: {img['caption']}\n\n"
            print(ret)
            return ret
        else:
            print("Error:", response.status_code, response.text)
            return ""
    except Exception as e:
        print(f"Error: {e}")
        return ""

@mcp.tool()
def cg_search_deep(query: str, max_results: int = 20) -> str:
    """Searches the Common Ground Knowledge Base using query rewriting and reranking.

    This function first rewrites the user's query using Gemini to extract keywords,
    then performs a hybrid search using both embedding search and BM25,
    and finally reranks the results for improved relevance.

    Args:
        query: The search query string.
        max_results: The maximum number of search results to return. Defaults to 20.

    Returns:
        A formatted string containing the search results (titles, content, URLs) and image results (URLs, captions).
    """
    return search(query, max_results, True, True)

@mcp.tool()
def cg_search(query: str, max_results: int = 20) -> str:
    """Performs a direct search on the Common Ground Knowledge Base.

    This function performs a hybrid search using both embedding search and BM25
    without query rewriting or result reranking.
    
    Args:
        query: The search query string.
        max_results: The maximum number of search results to return. Defaults to 20.
        
    Returns:
        A formatted string containing the search results (titles, content, URLs) and image results (URLs, captions).
    """
    return search(query, max_results, False, False)

if __name__ == "__main__":
    mcp.run()
