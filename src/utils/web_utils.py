import requests
import json
import re
import time
import logging
from bs4 import BeautifulSoup
from typing import Dict, List, Any, Optional, Tuple
from src.config.config import GOOGLE_API_KEY, GOOGLE_CX
import tiktoken  # Add tiktoken for token counting

# Add new imports for Selenium
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global driver variable to reuse browser instance
_driver = None

def get_chrome_driver(headless=True):
    global _driver
    try:
        if _driver is None:
            options = uc.ChromeOptions()
            if headless:
                options.add_argument('--headless=new')  # Tránh bị block
            options.add_argument('--disable-gpu')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-extensions')
            options.add_argument('--disable-popup-blocking')
            options.add_argument('--blink-settings=imagesEnabled=false')
            options.add_argument('--disable-blink-features=AutomationControlled')  # Bỏ flag tự động hóa
            options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')  # Fake User-Agent
            
            _driver = uc.Chrome(options=options)
            _driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")  # Xóa dấu hiệu WebDriver
            _driver.set_page_load_timeout(30)
            logger.info("Chrome driver initialized")
        return _driver
    except Exception as e:
        logger.error(f"Failed to initialize Chrome driver: {str(e)}")
        return None


def google_custom_search(query: str, num_results: int = 5, max_tokens: int = 4000) -> dict:
    """
    Perform a Google search using the Google Custom Search API and scrape content
    until reaching token limit.
    
    Args:
        query (str): The search query
        num_results (int): Number of results to return
        max_tokens (int): Maximum number of tokens for combined scraped content
        
    Returns:
        dict: Search results with metadata and combined scraped content
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
            'results': [],
            'combined_content': ""
        }
        
        if 'items' in search_results:
            # Extract all links first
            links = [item.get('link', '') for item in search_results['items']]
            
            # Scrape content from multiple links up to max_tokens
            combined_content, used_links = scrape_multiple_links(links, max_tokens)
            formatted_results['combined_content'] = combined_content
            
            # Process each search result
            for item in search_results['items']:
                result = {
                    'title': item.get('title', ''),
                    'link': item.get('link', ''),
                    'snippet': item.get('snippet', ''),
                    'date': item.get('pagemap', {}).get('metatags', [{}])[0].get('article:published_time', ''),
                    'used_for_content': item.get('link', '') in used_links
                }
                formatted_results['results'].append(result)
                
        return formatted_results
        
    except requests.exceptions.RequestException as e:
        return {
            'query': query,
            'error': f"Error during Google search: {str(e)}",
            'results': [],
            'combined_content': ""
        }

def scrape_multiple_links(urls: List[str], max_tokens: int = 4000) -> Tuple[str, List[str]]:
    """
    Scrape content from multiple URLs, stopping once token limit is reached.
    
    Args:
        urls (List[str]): List of URLs to scrape
        max_tokens (int): Maximum token count for combined content
        
    Returns:
        Tuple[str, List[str]]: Combined content and list of used URLs
    """
    combined_content = ""
    total_tokens = 0
    used_urls = []
    
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
    except:
        encoding = None
    
    for url in urls:
        # Skip empty URLs
        if not url:
            continue
            
        # Get content from this URL using enhanced scraping
        content, token_count = scrape_web_content_with_count(url, max_tokens=max_tokens, return_token_count=True)
        
        # Skip failed scrapes
        if content.startswith("Failed"):
            continue
            
        # Check if adding this content would exceed token limit
        if total_tokens + token_count > max_tokens:
            # If this is the first URL and it's too large, we need to truncate it
            if not combined_content:
                if encoding:
                    tokens = encoding.encode(content)
                    truncated_tokens = tokens[:max_tokens]
                    truncated_content = encoding.decode(truncated_tokens)
                    combined_content = f"{truncated_content}...\n[Content truncated due to token limit]"
                else:
                    # Fallback to character-based truncation
                    truncated_content = content[:max_tokens * 4]
                    combined_content = f"{truncated_content}...\n[Content truncated due to length]"
                used_urls.append(url)
            break
            
        # Add separator if not the first URL
        if combined_content:
            combined_content += f"\n\n--- Content from: {url} ---\n\n"
        else:
            combined_content += f"--- Content from: {url} ---\n\n"
            
        # Add content and update token count
        combined_content += content
        total_tokens += token_count
        used_urls.append(url)
        
        # If we've reached the token limit, stop
        if total_tokens >= max_tokens:
            break
            
    # If we didn't find any valid content
    if not combined_content:
        combined_content = "No valid content could be scraped from the provided URLs."
        
    return combined_content, used_urls

def scrape_with_selenium(url: str, timeout=30) -> str:
    """
    Scrape content from a webpage using Selenium with Undetected Chrome Driver
    
    Args:
        url (str): URL of the webpage to scrape
        timeout (int): Timeout in seconds for page load
        
    Returns:
        str: The scraped text content or error message
    """
    driver = get_chrome_driver()
    if not driver:
        return f"Failed to scrape {url}: Chrome driver not available"
    
    try:
        logger.info(f"Navigating to {url} with Selenium")
        driver.get(url)
        
        # Wait for page content to load
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Wait a bit for JavaScript to render content
        time.sleep(2)
        
        # Get page content
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(["script", "style", "header", "footer", "nav"]):
            element.extract()
        
        # Get text and clean it
        text = soup.get_text(separator='\n')
        
        # Clean up text: remove extra whitespace and empty lines
        lines = (line.strip() for line in text.splitlines())
        text = '\n'.join(line for line in lines if line)
        
        return text
        
    except TimeoutException:
        return f"Failed to scrape {url}: Page load timeout"
    except WebDriverException as e:
        return f"Failed to scrape {url}: Browser error - {str(e)}"
    except Exception as e:
        return f"Failed to process content from {url}: {str(e)}"

def scrape_web_content_with_count(url: str, max_tokens: int = 4000, return_token_count: bool = False) -> Any:
    """
    Scrape content from a webpage and return with token count if needed.
    Uses Selenium first, falls back to requests if Selenium fails.
    
    Args:
        url (str): URL of the webpage to scrape
        max_tokens (int): Maximum number of tokens to return
        return_token_count (bool): Whether to return token count with the content
        
    Returns:
        str or tuple: The scraped text content or (content, token_count)
    """
    if not url:
        return ("Failed to scrape: No URL provided.", 0) if return_token_count else "Failed to scrape: No URL provided."
    
    # Ignore URLs that are unlikely to be scrapable or might cause problems
    if any(x in url.lower() for x in ['.pdf', '.zip', '.jpg', '.png', '.mp3', '.mp4', 'youtube.com', 'youtu.be']):
        message = f"Failed to scrape: The URL {url} cannot be scraped (unsupported format)."
        return (message, 0) if return_token_count else message
    
    # Try with Selenium first
    content = scrape_with_selenium(url)
    
    # If Selenium failed, fall back to requests
    if content.startswith("Failed"):
        logger.info(f"Selenium scraping failed for {url}, falling back to requests")
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
            content = '\n'.join(line for line in lines if line)
        except requests.exceptions.RequestException as e:
            message = f"Failed to scrape {url}: {str(e)}"
            return (message, 0) if return_token_count else message
        except Exception as e:
            message = f"Failed to process content from {url}: {str(e)}"
            return (message, 0) if return_token_count else message
    
    # Count tokens
    token_count = 0
    try:
        # Use cl100k_base encoder which is used by most recent models
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(content)
        token_count = len(tokens)
        
        # Truncate if token count exceeds max_tokens
        if len(tokens) > max_tokens:
            truncated_tokens = tokens[:max_tokens]
            content = encoding.decode(truncated_tokens)
            content += "...\n[Content truncated due to token limit]"
            token_count = max_tokens
    except ImportError:
        # Fallback to character-based estimation
        token_count = len(content) // 4  # Rough estimate: 1 token ≈ 4 characters
        if len(content) > max_tokens * 4:
            content = content[:max_tokens * 4] + "...\n[Content truncated due to length]"
            token_count = max_tokens
    
    if return_token_count:
        return content, token_count
    return content

# Keep the original scrape_web_content function for backward compatibility
def scrape_web_content(url: str, max_tokens: int = 4000) -> str:
    """
    Scrape content from a webpage and limit by token count.
    
    Args:
        url (str): URL of the webpage to scrape
        max_tokens (int): Maximum number of tokens to return
        
    Returns:
        str: The scraped text content or error message
    """
    return scrape_web_content_with_count(url, max_tokens)

def cleanup_resources():
    """Close the browser when the application is shutting down"""
    global _driver
    if _driver:
        try:
            _driver.quit()
            _driver = None
            logger.info("Chrome driver closed")
        except Exception as e:
            logger.error(f"Error closing Chrome driver: {str(e)}")
