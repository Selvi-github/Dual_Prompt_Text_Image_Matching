"""
Image Retriever Module - No API Keys Required
Retrieves real images from the internet using web scraping
Works automatically without user configuration
"""

import requests
from PIL import Image
from io import BytesIO
import time
from typing import List, Dict
import logging
import re
from bs4 import BeautifulSoup
import urllib.parse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageRetriever:
    """
    Retrieves real images from the internet without requiring API keys
    Uses web scraping from multiple sources
    """
    
    def __init__(self, bing_api_key=None, google_api_key=None, google_cx=None):
        """
        Initialize ImageRetriever
        API keys are optional - system works without them using web scraping
        """
        self.bing_api_key = bing_api_key
        self.google_api_key = google_api_key
        self.google_cx = google_cx
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
    
    def retrieve_images(self, query: str, max_results: int = 8) -> List[Dict]:
        """
        Retrieve real images from the internet
        Works WITHOUT API keys using web scraping
        
        Args:
            query: Search query (incident description)
            max_results: Maximum number of images to retrieve
            
        Returns:
            List of dictionaries containing image data
        """
        logger.info(f"🔍 Searching for images: {query}")
        
        images = []
        
        # Try API methods first if keys are available
        if self.bing_api_key:
            logger.info("Using Bing API")
            images = self._retrieve_bing_images(query, max_results)
            if len(images) >= max_results:
                return images
        
        if self.google_api_key and self.google_cx:
            logger.info("Using Google API")
            api_images = self._retrieve_google_images(query, max_results - len(images))
            images.extend(api_images)
            if len(images) >= max_results:
                return images[:max_results]
        
        # Fall back to web scraping (NO API KEY REQUIRED)
        logger.info("🌐 Using web scraping (no API key needed)")
        
        # Try multiple sources
        sources = [
            self._scrape_duckduckgo,
            self._scrape_google_images,
            self._scrape_bing_images,
        ]
        
        for source_func in sources:
            try:
                scraped_images = source_func(query, max_results - len(images))
                images.extend(scraped_images)
                logger.info(f"✓ Got {len(scraped_images)} images from {source_func.__name__}")
                
                if len(images) >= max_results:
                    break
                    
            except Exception as e:
                logger.warning(f"⚠️ {source_func.__name__} failed: {e}")
                continue
        
        if images:
            logger.info(f"✅ Successfully retrieved {len(images)} real images")
        else:
            logger.error("❌ Could not retrieve images from any source")
        
        return images[:max_results]
    
    def _scrape_duckduckgo(self, query: str, max_results: int) -> List[Dict]:
        """
        Scrape images from DuckDuckGo (most reliable, no captcha)
        """
        logger.info("Trying DuckDuckGo...")
        
        images = []
        
        try:
            # DuckDuckGo image search endpoint
            url = "https://duckduckgo.com/"
            params = {
                'q': query,
                'iax': 'images',
                'ia': 'images'
            }
            
            session = requests.Session()
            
            # Get the main page first
            response = session.get(url, headers=self.headers, params=params, timeout=10)
            
            # Get the vqd token (required for DuckDuckGo API)
            vqd_match = re.search(r'vqd=([\d-]+)', response.text)
            if not vqd_match:
                logger.warning("Could not find vqd token")
                return []
            
            vqd = vqd_match.group(1)
            
            # Now query the actual image API
            api_url = "https://duckduckgo.com/i.js"
            api_params = {
                'l': 'us-en',
                'o': 'json',
                'q': query,
                'vqd': vqd,
                'f': ',,,',
                'p': '1'
            }
            
            api_response = session.get(api_url, headers=self.headers, params=api_params, timeout=10)
            data = api_response.json()
            
            # Extract images
            results = data.get('results', [])
            
            for item in results[:max_results]:
                try:
                    img_url = item.get('image')
                    if not img_url:
                        continue
                    
                    # Download image
                    img_response = requests.get(img_url, headers=self.headers, timeout=8)
                    img = Image.open(BytesIO(img_response.content)).convert('RGB')
                    
                    images.append({
                        'image': img,
                        'url': img_url,
                        'source': item.get('url', 'DuckDuckGo'),
                        'name': item.get('title', 'Image'),
                        'thumbnail': item.get('thumbnail', ''),
                        'date': 'Unknown'
                    })
                    
                    logger.info(f"✓ Downloaded: {item.get('title', 'Image')[:50]}")
                    time.sleep(0.2)
                    
                except Exception as e:
                    logger.debug(f"Failed to download image: {e}")
                    continue
            
            return images
            
        except Exception as e:
            logger.error(f"DuckDuckGo scraping failed: {e}")
            return []
    
    def _scrape_google_images(self, query: str, max_results: int) -> List[Dict]:
        """
        Scrape images from Google Images
        """
        logger.info("Trying Google Images...")
        
        images = []
        
        try:
            # Google Images search URL
            encoded_query = urllib.parse.quote(query)
            url = f"https://www.google.com/search?q={encoded_query}&tbm=isch"
            
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find image elements
            img_tags = soup.find_all('img')
            
            count = 0
            for img_tag in img_tags:
                if count >= max_results:
                    break
                
                try:
                    # Get image URL
                    img_url = img_tag.get('src') or img_tag.get('data-src')
                    
                    if not img_url or img_url.startswith('data:'):
                        continue
                    
                    # Download image
                    img_response = requests.get(img_url, headers=self.headers, timeout=8)
                    img = Image.open(BytesIO(img_response.content)).convert('RGB')
                    
                    # Filter out small images (likely icons/logos)
                    if img.size[0] < 100 or img.size[1] < 100:
                        continue
                    
                    images.append({
                        'image': img,
                        'url': img_url,
                        'source': 'Google Images',
                        'name': img_tag.get('alt', 'Image'),
                        'thumbnail': img_url,
                        'date': 'Unknown'
                    })
                    
                    count += 1
                    logger.info(f"✓ Downloaded from Google: {count}")
                    time.sleep(0.3)
                    
                except Exception as e:
                    logger.debug(f"Failed to download image: {e}")
                    continue
            
            return images
            
        except Exception as e:
            logger.error(f"Google Images scraping failed: {e}")
            return []
    
    def _scrape_bing_images(self, query: str, max_results: int) -> List[Dict]:
        """
        Scrape images from Bing Images (without API)
        """
        logger.info("Trying Bing Images...")
        
        images = []
        
        try:
            # Bing Images search URL
            encoded_query = urllib.parse.quote(query)
            url = f"https://www.bing.com/images/search?q={encoded_query}"
            
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find image links in Bing's format
            img_elements = soup.find_all('a', class_='iusc')
            
            count = 0
            for element in img_elements[:max_results * 2]:  # Get extra in case some fail
                if count >= max_results:
                    break
                
                try:
                    # Extract image URL from Bing's data attribute
                    m = element.get('m')
                    if not m:
                        continue
                    
                    # Parse JSON-like string
                    import json
                    data = json.loads(m)
                    img_url = data.get('murl') or data.get('turl')
                    
                    if not img_url:
                        continue
                    
                    # Download image
                    img_response = requests.get(img_url, headers=self.headers, timeout=8)
                    img = Image.open(BytesIO(img_response.content)).convert('RGB')
                    
                    # Filter out small images
                    if img.size[0] < 100 or img.size[1] < 100:
                        continue
                    
                    images.append({
                        'image': img,
                        'url': img_url,
                        'source': data.get('purl', 'Bing Images'),
                        'name': data.get('t', 'Image'),
                        'thumbnail': data.get('turl', ''),
                        'date': 'Unknown'
                    })
                    
                    count += 1
                    logger.info(f"✓ Downloaded from Bing: {count}")
                    time.sleep(0.3)
                    
                except Exception as e:
                    logger.debug(f"Failed to download image: {e}")
                    continue
            
            return images
            
        except Exception as e:
            logger.error(f"Bing Images scraping failed: {e}")
            return []
    
    def _retrieve_bing_images(self, query: str, max_results: int) -> List[Dict]:
        """Retrieve images using Bing API (if key provided)"""
        if not self.bing_api_key:
            return []
        
        endpoint = "https://api.bing.microsoft.com/v7.0/images/search"
        headers = {'Ocp-Apim-Subscription-Key': self.bing_api_key}
        params = {
            'q': query,
            'count': max_results,
            'imageType': 'Photo',
            'safeSearch': 'Moderate',
            'freshness': 'Month'
        }
        
        try:
            response = requests.get(endpoint, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            images = []
            
            for item in data.get('value', [])[:max_results]:
                try:
                    img_response = requests.get(item['contentUrl'], timeout=10, headers=self.headers)
                    img = Image.open(BytesIO(img_response.content)).convert('RGB')
                    
                    images.append({
                        'image': img,
                        'url': item['contentUrl'],
                        'source': item.get('hostPageUrl', 'Unknown'),
                        'name': item.get('name', 'Untitled'),
                        'thumbnail': item.get('thumbnailUrl', ''),
                        'date': item.get('datePublished', 'Unknown')
                    })
                    time.sleep(0.1)
                except:
                    continue
            
            return images
        except:
            return []
    
    def _retrieve_google_images(self, query: str, max_results: int) -> List[Dict]:
        """Retrieve images using Google API (if key provided)"""
        if not self.google_api_key or not self.google_cx:
            return []
        
        endpoint = "https://www.googleapis.com/customsearch/v1"
        params = {
            'key': self.google_api_key,
            'cx': self.google_cx,
            'q': query,
            'searchType': 'image',
            'num': min(max_results, 10)
        }
        
        try:
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            images = []
            
            for item in data.get('items', [])[:max_results]:
                try:
                    img_response = requests.get(item['link'], timeout=10, headers=self.headers)
                    img = Image.open(BytesIO(img_response.content)).convert('RGB')
                    
                    images.append({
                        'image': img,
                        'url': item['link'],
                        'source': item.get('displayLink', 'Unknown'),
                        'name': item.get('title', 'Untitled'),
                        'thumbnail': item.get('image', {}).get('thumbnailLink', ''),
                        'date': 'Unknown'
                    })
                    time.sleep(0.1)
                except:
                    continue
            
            return images
        except:
            return []
    
    def get_demo_images(self, query: str, max_results: int) -> List[Dict]:
        """
        This method now redirects to web scraping
        NO DEMO/PLACEHOLDER IMAGES ANYMORE
        """
        return self.retrieve_images(query, max_results)


# Test the scraper
if __name__ == "__main__":
    print("🔍 Testing Image Retriever (No API Keys Required)...")
    
    retriever = ImageRetriever()
    
    # Test queries
    test_queries = [
        "Chennai floods 2024",
        "Vijay TVK rally Tamil Nadu",
        "Mumbai fire incident 2024"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Testing: {query}")
        print('='*60)
        
        images = retriever.retrieve_images(query, max_results=3)
        print(f"✅ Retrieved {len(images)} real images")
        
        for i, img_data in enumerate(images, 1):
            print(f"{i}. {img_data['name'][:50]}... from {img_data['source'][:30]}")