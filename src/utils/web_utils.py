import requests
import json
import re
from bs4 import BeautifulSoup
from typing import Dict, List, Any, Optional
from src.config.config import GOOGLE_API_KEY, GOOGLE_CX
import tiktoken  # Add tiktoken for token counting

def google_custom_search(query: str, num_results: int = 3, auto_scrape: bool = True) -> dict:
    """
    Perform a Google search using the Google Custom Search API.
    
    Args:
        query (str): The search query
        num_results (int): Number of results to return
        auto_scrape (bool): Whether to automatically scrape top results
        
    Returns:
        dict: Search results with metadata and optional scraped content
    """
    try:
        search_url = f"https://www.googleapis.com/customsearch/v1"
        params = {
            'key': GOOGLE_API_KEY,
            'cx': GOOGLE_CX,
            'q': query,
            'num': min(num_results, 10)  # Google API maximum is 10
        }
        
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        search_results = response.json()
        
        # Format the results for ease of use
        formatted_results = {
            'query': query,
            'results': []
        }
        
        if 'items' in search_results:
            for item in search_results['items']:
                result = {
                    'title': item.get('title', ''),
                    'link': item.get('link', ''),
                    'snippet': item.get('snippet', ''),
                    'date': item.get('pagemap', {}).get('metatags', [{}])[0].get('article:published_time', '')
                }
                
                # Auto-scrape the content if requested
                if auto_scrape:
                    content = scrape_web_content(result['link'])
                    if not content.startswith('Failed'):
                        result['scraped_content'] = content
                        
                formatted_results['results'].append(result)
                
        return formatted_results
        
    except requests.exceptions.RequestException as e:
        return {
            'query': query,
            'error': f"Error during Google search: {str(e)}",
            'results': []
        }

def scrape_web_content(url: str, max_tokens: int = 5000) -> str:
    """
    Scrape content from a webpage and limit by token count.
    
    Args:
        url (str): URL of the webpage to scrape
        max_tokens (int): Maximum number of tokens to return
        
    Returns:
        str: The scraped text content or error message
    """
    if not url:
        return "Failed to scrape: No URL provided."
    
    # Ignore URLs that are unlikely to be scrapable or might cause problems
    if any(x in url.lower() for x in ['.pdf', '.zip', '.jpg', '.png', '.mp3', '.mp4', 'youtube.com', 'youtu.be']):
        return f"Failed to scrape: The URL {url} cannot be scraped (unsupported format)."
    
    try:
        # Add user agent to mimic a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse the content with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "header", "footer", "nav"]):
            script.extract()
            
        # Get the text content
        text = soup.get_text(separator='\n')
        
        # Clean up text: remove extra whitespace and empty lines
        lines = (line.strip() for line in text.splitlines())
        text = '\n'.join(line for line in lines if line)
        
        # Count tokens and truncate if needed
        try:
            # Use cl100k_base encoder which is used by most recent models
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(text)
            
            # Truncate if token count exceeds max_tokens
            if len(tokens) > max_tokens:
                # Truncate to max_tokens
                truncated_tokens = tokens[:max_tokens]
                text = encoding.decode(truncated_tokens)
                text += "...\n[Content truncated due to token limit]"
        except ImportError:
            # Fallback to character-based truncation if tiktoken is not available
            if len(text) > max_tokens * 4:  # Rough estimate: 1 token ≈ 4 characters
                text = text[:max_tokens * 4] + "...\n[Content truncated due to length]"
            
        return text
        
    except requests.exceptions.RequestException as e:
        return f"Failed to scrape {url}: {str(e)}"
    except Exception as e:
        return f"Failed to process content from {url}: {str(e)}"
