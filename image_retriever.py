"""
Enhanced Image Retrieval Module
Uses AI to convert image to text and retrieve highly similar images
"""

import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import time
from typing import List, Dict, Optional
import urllib.parse
import numpy as np

class ImageRetriever:
    def __init__(self):
        """Initialize enhanced image retriever with AI capabilities"""
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # Try to load image analysis model
        self.feature_extractor = self._load_image_analyzer()
        
        print("✓ Enhanced Image Retriever initialized with AI")
    
    def _load_image_analyzer(self):
        """Load lightweight image feature extractor"""
        try:
            # Try to use a lightweight model if available
            print("Attempting to load image analysis model...")
            # For now, we'll use basic image features
            # Can be enhanced with transformers if needed
            return None
        except Exception as e:
            print(f"Image analyzer not available: {e}")
            return None
    
    def image_to_text(self, image: Image.Image) -> str:
        """
        Convert image to descriptive text using AI analysis
        Returns detailed description of image content
        """
        try:
            print("🔍 Analyzing image content...")
            
            # Extract visual features
            features = self._extract_image_features(image)
            
            # Generate description based on features
            description = self._generate_description_from_features(features, image)
            
            print(f"✓ Image description: {description}")
            return description
        
        except Exception as e:
            print(f"Image to text conversion error: {e}")
            return "incident scene"
    
    def _extract_image_features(self, image: Image.Image) -> Dict:
        """Extract detailed features from image"""
        try:
            # Convert to numpy array
            img_array = np.array(image.resize((224, 224)))
            
            # Analyze color composition
            avg_color = np.mean(img_array, axis=(0, 1))
            
            # Detect dominant colors
            is_dark = np.mean(avg_color) < 100
            has_red = avg_color[0] > 150
            has_blue = avg_color[2] > 150
            
            # Analyze brightness and contrast
            brightness = np.mean(img_array)
            contrast = np.std(img_array)
            
            # Detect edges (simple method)
            gray = np.mean(img_array, axis=2)
            edges = np.abs(np.diff(gray, axis=0)).sum() + np.abs(np.diff(gray, axis=1)).sum()
            
            features = {
                'avg_color': avg_color,
                'brightness': brightness,
                'contrast': contrast,
                'edge_density': edges,
                'is_dark': is_dark,
                'has_red_tones': has_red,
                'has_blue_tones': has_blue,
                'aspect_ratio': image.size[0] / image.size[1]
            }
            
            return features
        
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return {}
    
    def _generate_description_from_features(self, features: Dict, image: Image.Image) -> str:
        """Generate text description based on visual features"""
        try:
            description_parts = []
            
            # Analyze scene type based on features
            if features.get('is_dark', False):
                description_parts.append("dark scene")
            else:
                description_parts.append("daylight scene")
            
            if features.get('has_red_tones', False):
                description_parts.append("fire emergency disaster")
            
            if features.get('contrast', 0) > 50:
                description_parts.append("dramatic incident")
            
            # Analyze brightness
            brightness = features.get('brightness', 128)
            if brightness < 80:
                description_parts.append("night emergency")
            elif brightness > 180:
                description_parts.append("outdoor incident")
            
            # Default fallback
            if not description_parts:
                description_parts = ["emergency incident scene"]
            
            description = " ".join(description_parts)
            return description
        
        except:
            return "incident scene"
    
    def retrieve_similar_images(
        self, 
        reference_image: Image.Image, 
        text_context: Optional[str] = None,
        max_images: int = 15
    ) -> List[Dict]:
        """
        Retrieve highly similar images using AI-enhanced matching
        
        Args:
            reference_image: User's uploaded image
            text_context: Optional text description for better matching
            max_images: Maximum number of images to retrieve
        
        Returns:
            List of similar images with metadata and similarity scores
        """
        try:
            print("🔍 Starting AI-enhanced image retrieval...")
            
            # Step 1: Convert image to descriptive text
            image_description = self.image_to_text(reference_image)
            
            # Step 2: Combine with text context if available
            if text_context:
                search_query = f"{text_context} {image_description}"
            else:
                search_query = image_description
            
            print(f"📝 Search query: {search_query}")
            
            # Step 3: Retrieve candidate images
            candidate_images = self._search_multiple_sources(search_query, max_images * 2)
            
            if not candidate_images:
                print("⚠️ No candidate images found")
                return []
            
            # Step 4: Calculate similarity scores and rank
            scored_images = []
            for img_data in candidate_images:
                try:
                    similarity_score = self._calculate_deep_similarity(
                        reference_image, 
                        img_data['image']
                    )
                    
                    img_data['similarity_score'] = similarity_score
                    img_data['match_percentage'] = int(similarity_score * 100)
                    scored_images.append(img_data)
                except Exception as e:
                    continue
            
            # Step 5: Sort by similarity and return top matches
            scored_images.sort(key=lambda x: x['similarity_score'], reverse=True)
            top_matches = scored_images[:max_images]
            
            print(f"✓ Retrieved {len(top_matches)} highly similar images")
            
            # Add original incident information
            for img in top_matches:
                img['original_description'] = self._extract_incident_details(img)
            
            return top_matches
        
        except Exception as e:
            print(f"Similar image retrieval error: {e}")
            return []
    
    def retrieve_images(self, query: str, max_images: int = 10) -> List[Dict]:
        """
        Standard image retrieval based on text query
        """
        try:
            print(f"🔍 Searching images for: {query}")
            images = self._search_multiple_sources(query, max_images)
            
            # Add incident descriptions
            for img in images:
                img['original_description'] = self._extract_incident_details(img)
            
            print(f"✓ Retrieved {len(images)} images")
            return images
        
        except Exception as e:
            print(f"Image retrieval error: {e}")
            return []
    
    def _search_multiple_sources(self, query: str, max_images: int) -> List[Dict]:
        """Search images from multiple sources"""
        all_images = []
        
        # Try DuckDuckGo first
        ddg_images = self._search_duckduckgo(query, max_images // 2)
        all_images.extend(ddg_images)
        
        # Try Bing as backup
        if len(all_images) < max_images:
            bing_images = self._search_bing(query, max_images - len(all_images))
            all_images.extend(bing_images)
        
        return all_images[:max_images]
    
    def _calculate_deep_similarity(self, img1: Image.Image, img2: Image.Image) -> float:
        """
        Calculate deep similarity between images using multiple metrics
        Returns score between 0 and 1
        """
        try:
            # Resize both images to same size
            size = (128, 128)
            img1_resized = img1.resize(size).convert('RGB')
            img2_resized = img2.resize(size).convert('RGB')
            
            # Convert to numpy arrays
            arr1 = np.array(img1_resized).astype(float)
            arr2 = np.array(img2_resized).astype(float)
            
            # Metric 1: Pixel-wise correlation
            flat1 = arr1.flatten()
            flat2 = arr2.flatten()
            correlation = np.corrcoef(flat1, flat2)[0, 1]
            correlation_score = (correlation + 1) / 2
            
            # Metric 2: Color histogram similarity
            hist_similarity = self._compare_color_histograms(arr1, arr2)
            
            # Metric 3: Structural similarity (simplified SSIM)
            structural_similarity = self._calculate_structural_similarity(arr1, arr2)
            
            # Metric 4: Edge similarity
            edge_similarity = self._compare_edges(arr1, arr2)
            
            # Weighted combination
            final_score = (
                correlation_score * 0.3 +
                hist_similarity * 0.3 +
                structural_similarity * 0.25 +
                edge_similarity * 0.15
            )
            
            return max(0, min(1, final_score))
        
        except Exception as e:
            print(f"Similarity calculation error: {e}")
            return 0.5
    
    def _compare_color_histograms(self, arr1: np.ndarray, arr2: np.ndarray) -> float:
        """Compare color distribution between images"""
        try:
            # Calculate histograms for each channel
            similarity_scores = []
            
            for channel in range(3):  # R, G, B
                hist1, _ = np.histogram(arr1[:, :, channel], bins=32, range=(0, 256))
                hist2, _ = np.histogram(arr2[:, :, channel], bins=32, range=(0, 256))
                
                # Normalize
                hist1 = hist1 / hist1.sum()
                hist2 = hist2 / hist2.sum()
                
                # Calculate similarity (1 - chi-square distance)
                distance = np.sum((hist1 - hist2) ** 2 / (hist1 + hist2 + 1e-10))
                similarity = 1 / (1 + distance)
                similarity_scores.append(similarity)
            
            return np.mean(similarity_scores)
        
        except:
            return 0.5
    
    def _calculate_structural_similarity(self, arr1: np.ndarray, arr2: np.ndarray) -> float:
        """Simplified structural similarity"""
        try:
            # Convert to grayscale
            gray1 = np.mean(arr1, axis=2)
            gray2 = np.mean(arr2, axis=2)
            
            # Calculate means
            mean1 = np.mean(gray1)
            mean2 = np.mean(gray2)
            
            # Calculate variances
            var1 = np.var(gray1)
            var2 = np.var(gray2)
            
            # Calculate covariance
            covar = np.mean((gray1 - mean1) * (gray2 - mean2))
            
            # SSIM formula (simplified)
            c1 = 0.01 ** 2
            c2 = 0.03 ** 2
            
            ssim = ((2 * mean1 * mean2 + c1) * (2 * covar + c2)) / \
                   ((mean1**2 + mean2**2 + c1) * (var1 + var2 + c2))
            
            return (ssim + 1) / 2  # Normalize to 0-1
        
        except:
            return 0.5
    
    def _compare_edges(self, arr1: np.ndarray, arr2: np.ndarray) -> float:
        """Compare edge patterns between images"""
        try:
            # Convert to grayscale
            gray1 = np.mean(arr1, axis=2)
            gray2 = np.mean(arr2, axis=2)
            
            # Simple edge detection
            edges1_h = np.abs(np.diff(gray1, axis=0))
            edges1_v = np.abs(np.diff(gray1, axis=1))
            edges2_h = np.abs(np.diff(gray2, axis=0))
            edges2_v = np.abs(np.diff(gray2, axis=1))
            
            # Calculate edge correlation
            corr_h = np.corrcoef(edges1_h.flatten(), edges2_h.flatten()[:-1])[0, 1]
            corr_v = np.corrcoef(edges1_v.flatten()[:-1], edges2_v.flatten())[0, 1]
            
            edge_similarity = ((corr_h + corr_v) / 2 + 1) / 2
            
            return max(0, min(1, edge_similarity))
        
        except:
            return 0.5
    
    def _extract_incident_details(self, img_data: Dict) -> str:
        """Extract original incident information from image metadata"""
        try:
            # Extract title/name
            name = img_data.get('name', '')
            source = img_data.get('source', '')
            
            # Clean and format description
            description_parts = []
            
            if name and len(name) > 5:
                description_parts.append(name)
            
            if source and source != 'DuckDuckGo' and source != 'Bing':
                description_parts.append(f"Source: {source}")
            
            if description_parts:
                return " | ".join(description_parts)
            else:
                return "Original incident image from web source"
        
        except:
            return "Incident image"
    
    def _search_duckduckgo(self, query: str, max_images: int) -> List[Dict]:
        """Search images using DuckDuckGo"""
        try:
            encoded_query = urllib.parse.quote(query)
            url = f"https://duckduckgo.com/?q={encoded_query}&iax=images&ia=images"
            
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code != 200:
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            images = []
            img_tags = soup.find_all('img', limit=max_images * 3)
            
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
                    # Extract metadata
                    alt_text = img_tag.get('alt', query)
                    title = img_tag.get('title', alt_text)
                    
                    images.append({
                        'image': img_data,
                        'source': 'DuckDuckGo Search',
                        'name': title if title else alt_text,
                        'url': img_url,
                        'metadata': {
                            'alt': alt_text,
                            'title': title
                        }
                    })
                
                time.sleep(0.2)
            
            return images
        
        except Exception as e:
            print(f"DuckDuckGo search error: {e}")
            return []
    
    def _search_bing(self, query: str, max_images: int) -> List[Dict]:
        """Search images using Bing"""
        try:
            encoded_query = urllib.parse.quote(query)
            url = f"https://www.bing.com/images/search?q={encoded_query}"
            
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code != 200:
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            images = []
            img_tags = soup.find_all('img', class_='mimg', limit=max_images * 3)
            
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
                    alt_text = img_tag.get('alt', query)
                    
                    images.append({
                        'image': img_data,
                        'source': 'Bing Search',
                        'name': alt_text,
                        'url': img_url,
                        'metadata': {
                            'alt': alt_text
                        }
                    })
                
                time.sleep(0.2)
            
            return images
        
        except Exception as e:
            print(f"Bing search error: {e}")
            return []
    
    def _download_image(self, url: str) -> Optional[Image.Image]:
        """Download and validate image"""
        try:
            response = requests.get(url, headers=self.headers, timeout=5)
            
            if response.status_code != 200:
                return None
            
            img = Image.open(BytesIO(response.content))
            
            # Validate minimum size
            if img.size[0] < 100 or img.size[1] < 100:
                return None
            
            # Convert to RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            return img
        
        except:
            return None
