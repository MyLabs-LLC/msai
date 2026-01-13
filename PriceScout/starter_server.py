
import os
import json
import logging
from typing import List, Dict, Optional
from firecrawl import FirecrawlApp
from urllib.parse import urlparse
from datetime import datetime
from mcp.server.fastmcp import FastMCP

from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SCRAPE_DIR = "scraped_content"

mcp = FastMCP("llm_inference")

@mcp.tool()
def scrape_websites(
    websites: Dict[str, str],
    formats: List[str] = ['markdown', 'html'],
    api_key: Optional[str] = None
) -> List[str]:
    """
    Scrape multiple websites using Firecrawl and store their content.
    
    Args:
        websites: Dictionary of provider_name -> URL mappings
        formats: List of formats to scrape ['markdown', 'html'] (default: both)
        api_key: Firecrawl API key (if None, expects environment variable)
        
    Returns:
        List of provider names for successfully scraped websites
    """
    
    if api_key is None:
        api_key = os.getenv('FIRECRAWL_API_KEY')
        if not api_key:
            raise ValueError("API key must be provided or set as FIRECRAWL_API_KEY environment variable")
    
    app = FirecrawlApp(api_key=api_key)
    
    path = os.path.join(SCRAPE_DIR)
    os.makedirs(path, exist_ok=True)
    
    # save the scraped content to files and then create scraped_metadata.json as a summary file
    # check if the provider has already been scraped and decide if you want to overwrite
    # {
    #     "cloudrift_ai": {
    #         "provider_name": "cloudrift_ai",
    #         "url": "https://www.cloudrift.ai/inference",
    #         "domain": "www.cloudrift.ai",
    #         "scraped_at": "2025-10-23T00:44:59.902569",
    #         "formats": [
    #             "markdown",
    #             "html"
    #         ],
    #         "success": "true",
    #         "content_files": {
    #             "markdown": "cloudrift_ai_markdown.txt",
    #             "html": "cloudrift_ai_html.txt"
    #         },
    #         "title": "AI Inference",
    #         "description": "Scraped content goes here"
    #     }
    # }
    metadata_file = os.path.join(path, "scraped_metadata.json")

    # Load existing metadata or initialize empty dict
    scraped_metadata = {}
    try:
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                content = f.read().strip()
                if content:
                    scraped_metadata = json.loads(content)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Could not load existing metadata: {e}")
        scraped_metadata = {}

    successful_scrapes = []

    # Loop through websites and scrape each one
    for provider_name, url in websites.items():
        try:
            logger.info(f"Scraping {provider_name}: {url}")
            
            # Call Firecrawl to scrape the URL (using scrape() method, not scrape_url())
            # Convert format strings to FormatOption if needed
            format_list = formats if isinstance(formats, list) else [formats]
            scrape_result = app.scrape(url, formats=format_list)
            result_dict = scrape_result.model_dump() if hasattr(scrape_result, 'model_dump') else scrape_result
            
            # Parse domain from URL
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            
            # Create metadata entry for this provider
            metadata = {
                "provider_name": provider_name,
                "url": url,
                "domain": domain,
                "scraped_at": datetime.now().isoformat(),
                "formats": formats,
                "success": False,
                "content_files": {},
                "title": "",
                "description": ""
            }
            
            # Check if scraping was successful (Document object has content fields)
            if scrape_result and (hasattr(scrape_result, 'markdown') or hasattr(scrape_result, 'html')):
                # Save content for each format
                for format_type in formats:
                    # Access content from Document object
                    content = None
                    if format_type == 'markdown' and hasattr(scrape_result, 'markdown'):
                        content = scrape_result.markdown
                    elif format_type == 'html' and hasattr(scrape_result, 'html'):
                        content = scrape_result.html
                    elif format_type == 'raw_html' and hasattr(scrape_result, 'raw_html'):
                        content = scrape_result.raw_html
                    
                    if content:
                        filename = f"{provider_name}_{format_type}.txt"
                        filepath = os.path.join(path, filename)
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(content)
                        metadata["content_files"][format_type] = filename
                        logger.info(f"Saved {format_type} content to {filename}")
                
                # Update metadata with page info
                metadata["success"] = True
                if hasattr(scrape_result, 'metadata') and scrape_result.metadata:
                    metadata_obj = scrape_result.metadata
                    if hasattr(metadata_obj, 'title'):
                        metadata["title"] = metadata_obj.title or ""
                    elif isinstance(metadata_obj, dict):
                        metadata["title"] = metadata_obj.get('title', '')
                    else:
                        metadata["title"] = str(metadata_obj) if metadata_obj else ""
                    
                    if hasattr(metadata_obj, 'description'):
                        metadata["description"] = metadata_obj.description or ""
                    elif isinstance(metadata_obj, dict):
                        metadata["description"] = metadata_obj.get('description', '')
                else:
                    metadata["title"] = ""
                    metadata["description"] = ""
                
                successful_scrapes.append(provider_name)
                logger.info(f"Successfully scraped {provider_name}")
            else:
                error_msg = "No content returned from scrape"
                logger.error(f"Failed to scrape {provider_name}: {error_msg}")
                metadata["error"] = error_msg
            
            # Add to metadata dictionary
            scraped_metadata[provider_name] = metadata
            
        except Exception as e:
            logger.error(f"Error scraping {provider_name}: {e}", exc_info=True)
            scraped_metadata[provider_name] = {
                "provider_name": provider_name,
                "url": url,
                "domain": urlparse(url).netloc,
                "scraped_at": datetime.now().isoformat(),
                "formats": formats,
                "success": False,
                "error": str(e)
            }

    # Write metadata to file
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(scraped_metadata, f, indent=2)
    
    logger.info(f"Scraping complete. Successfully scraped {len(successful_scrapes)}/{len(websites)} websites")
    return successful_scrapes

@mcp.tool()
def extract_scraped_info(identifier: str) -> str:
    """
    Extract information about a scraped website.
    
    Args:
        identifier: The provider name, full URL, or domain to look for
        
    Returns:
        Formatted JSON string with the scraped information
    """
    
    logger.info(f"Extracting information for identifier: {identifier}")
    logger.info(f"Files in {SCRAPE_DIR}: {os.listdir(SCRAPE_DIR)}")

    metadata_file = os.path.join(SCRAPE_DIR, "scraped_metadata.json")
    logger.info(f"Checking metadata file: {metadata_file}")

    try:
        # Load metadata file
        with open(metadata_file, 'r', encoding='utf-8') as f:
            scraped_metadata = json.load(f)
        
        # Search for a match
        for provider_name, metadata in scraped_metadata.items():
            # Check if identifier matches provider name, URL, or domain
            if (identifier.lower() == provider_name.lower() or
                identifier.lower() in metadata.get('url', '').lower() or
                identifier.lower() in metadata.get('domain', '').lower() or
                provider_name.lower() in identifier.lower()):
                
                logger.info(f"Found match for identifier '{identifier}': {provider_name}")
                
                # Make a copy of metadata to return
                result = metadata.copy()
                
                # If there are content files, read their content
                if 'content_files' in metadata and metadata['content_files']:
                    result['content'] = {}
                    for format_type, filename in metadata['content_files'].items():
                        filepath = os.path.join(SCRAPE_DIR, filename)
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                result['content'][format_type] = f.read()
                        except IOError as e:
                            logger.warning(f"Could not read content file {filename}: {e}")
                            result['content'][format_type] = f"Error reading file: {e}"
                
                return json.dumps(result, indent=2)
        
        # No match found
        logger.info(f"No match found for identifier: {identifier}")
        return f"There's no saved information related to identifier '{identifier}'."
        
    except FileNotFoundError:
        logger.warning(f"Metadata file not found: {metadata_file}")
        return f"No scraped data available. Please scrape websites first."
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing metadata file: {e}")
        return f"Error reading metadata: {e}"
    except Exception as e:
        logger.error(f"Error extracting info: {e}")
        return f"Error extracting information: {e}"

if __name__ == "__main__":
    mcp.run(transport="stdio")