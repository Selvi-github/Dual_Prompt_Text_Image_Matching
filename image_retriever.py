"""
Image Retrieval Module
Retrieves images from web using BeautifulSoup (NO API NEEDED)
"""

import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import time
from typing import List, Dict
import urllib.parse

class ImageRetriever:
    def __init__(self):
        """Initialize image retriever"""
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        print("✓ Image Retriever initialized")
    
    def retrieve_images(self, query: str, max_images: int = 10) -> List[Dict]:
        """
        Retrieve images from web based on query
        Returns list of dicts with 'image', 'source', 'name'
        """
        try:
            print(f"Searching images for: {query}")
            
            # Use DuckDuckGo image search (no API needed)
            images = self._search_duckduckgo(query, max_images)
            
            if not images:
                # Fallback to Bing
                images = self._search_bing(query, max_images)
            
            print(f"Retrieved {len(images)} images")
            return images
        
        except Exception as e:
            print(f"Image retrieval error: {e}")
            return []
    
    def _search_duckduckgo(self, query: str, max_images: int) -> List[Dict]:
        """Search images using DuckDuckGo"""
        try:
            # Encode query
            encoded_query = urllib.parse.quote(query)
            url = f"https://duckduckgo.com/?q={encoded_query}&iax=images&ia=images"
            
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code != 200:
                return []
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            images = []
            img_tags = soup.find_all('img', limit=max_images * 2)
            
            for img_tag in img_tags:
                if len(images) >= max_images:
                    break
                
                # Get image URL
                img_url = img_tag.get('src') or img_tag.get('data-src')
                
                if not img_url or img_url.startswith('data:'):
                    continue
                
                # Ensure full URL
                if img_url.startswith('//'):
                    img_url = 'https:' + img_url
                elif not img_url.startswith('http'):
                    continue
                
                # Download image
                img_data = self._download_image(img_url)
                
                if img_data:
                    images.append({
                        'image': img_data,
                        'source': 'DuckDuckGo',
                        'name': query,
                        'url': img_url
                    })
                
                time.sleep(0.2)  # Rate limiting
            
            return images
        
        except Exception as e:
            print(f"DuckDuckGo search error: {e}")
            return []
    
    def _search_bing(self, query: str, max_images: int) -> List[Dict]:
        """Search images using Bing (fallback)"""
        try:
            encoded_query = urllib.parse.quote(query)
            url = f"https://www.bing.com/images/search?q={encoded_query}"
            
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code != 200:
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            images = []
            img_tags = soup.find_all('img', class_='mimg', limit=max_images * 2)
            
            for img_tag in img_tags:
                if len(images) >= max_images:
                    break
                
                img_url = img_tag.get('src') or img_tag.get('data-src')
                
                if not img_url or img_url.startswith('data:'):
                    continue
                
                if img_url.startswith('//'):
                    img_url = 'https:' + img_url
                elif not img_url.startswith('http'):
                    continue
                
                img_data = self._download_image(img_url)
                
                if img_data:
                    images.append({
                        'image': img_data,
                        'source': 'Bing',
                        'name': query,
                        'url': img_url
                    })
                
                time.sleep(0.2)
            
            return images
        
        except Exception as e:
            print(f"Bing search error: {e}")
            return []
    
    def _download_image(self, url: str) -> Image.Image:
        """Download and validate image"""
        try:
            response = requests.get(url, headers=self.headers, timeout=5)
            
            if response.status_code != 200:
                return None
            
            # Try to open as image
            img = Image.open(BytesIO(response.content))
            
            # Validate image
            if img.size[0] < 100 or img.size[1] < 100:
                return None
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            return img
        
        except Exception as e:
            return None
    
    def get_sample_images(self, count: int = 5) -> List[Dict]:
        """Get sample placeholder images for testing"""
        images = []
        
        # Create simple colored placeholder images
        for i in range(count):
            img = Image.new('RGB', (300, 200), color=(100 + i * 30, 150, 200))
            images.append({
                'image': img,
                'source': 'Placeholder',
                'name': f'Sample {i+1}',
                'url': 'N/A'
            })
        
        return images
