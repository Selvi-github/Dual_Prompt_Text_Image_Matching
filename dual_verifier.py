"""
Verification Module - FIXED VERSION
Uses CLIP correctly for accurate image-text matching
"""

import torch
import clip
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from typing import List, Dict
import numpy as np

class IncidentVerifier:
    def __init__(self, device: str = None):
        """Initialize CLIP and BLIP-2 models"""
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading models on {self.device}...")
        
        # Load CLIP for image-text similarity
        try:
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            print("✓ CLIP model loaded")
        except Exception as e:
            print(f"CLIP loading error: {e}")
            self.clip_model = None
        
        # Load BLIP-2 for image captioning
        try:
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            ).to(self.device)
            print("✓ BLIP model loaded")
        except Exception as e:
            print(f"BLIP loading error: {e}")
            self.blip_model = None
    
    def generate_caption(self, image: Image.Image) -> str:
        """Generate caption for an image using BLIP-2"""
        try:
            if self.blip_model is None:
                return "Caption generation unavailable"
            
            inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                out = self.blip_model.generate(**inputs, max_length=50)
            
            caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            print(f"Caption generation error: {e}")
            return "Unable to generate caption"
    
    def compute_clip_similarity(self, image: Image.Image, text: str) -> float:
        """
        FIXED: Compute accurate CLIP similarity between image and text
        Returns value between 0 and 1
        """
        try:
            if self.clip_model is None:
                return 0.5
            
            # Preprocess image
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            
            # Tokenize text
            text_input = clip.tokenize([text], truncate=True).to(self.device)
            
            # Compute features
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_input)
                
                # Normalize features (IMPORTANT!)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Compute cosine similarity
                similarity = (image_features @ text_features.T).squeeze()
                
                # Convert to 0-1 range using sigmoid for better scaling
                # CLIP raw scores are usually between -1 to 1, we normalize
                similarity = (similarity + 1) / 2  # Convert [-1,1] to [0,1]
            
            return float(similarity.cpu())
            
        except Exception as e:
            print(f"CLIP similarity error: {e}")
            return 0.5
    
    def verify_text_to_image(
        self, 
        text: str, 
        retrieved_images: List[Dict]
    ) -> Dict:
        """
        FIXED: Verify incident by matching text with retrieved images
        Uses proper CLIP similarity thresholds
        """
        try:
            if not retrieved_images:
                return {
                    'authenticity': 'UNCERTAIN',
                    'confidence': 0,
                    'explanation': 'No images found for verification',
                    'top_matches': []
                }
            
            similarities = []
            
            # Calculate similarity for each retrieved image
            for img_data in retrieved_images:
                try:
                    img = img_data['image']
                    score = self.compute_clip_similarity(img, text)
                    
                    similarities.append({
                        'image': img,
                        'score': score,
                        'name': img_data['name'],
                        'url': img_data['url'],
                        'source': img_data['source']
                    })
                    
                    print(f"  • {img_data['name'][:40]}: {score:.4f}")
                    
                except Exception as e:
                    print(f"Error processing image: {e}")
                    continue
            
            if not similarities:
                return {
                    'authenticity': 'UNCERTAIN',
                    'confidence': 0,
                    'explanation': 'Unable to process retrieved images',
                    'top_matches': []
                }
            
            # Sort by similarity score (highest first)
            similarities.sort(key=lambda x: x['score'], reverse=True)
            
            # Get top 5 matches
            top_matches = similarities[:5]
            
            # Calculate statistics
            top_3_scores = [m['score'] for m in similarities[:3]]
            avg_top3 = np.mean(top_3_scores)
            max_score = similarities[0]['score']
            
            print(f"\n📊 Scores - Max: {max_score:.4f} | Avg(top-3): {avg_top3:.4f}")
            
            # FIXED THRESHOLDS based on normalized CLIP scores
            # After normalization, scores are 0-1:
            # 0.65+ = Very strong match (REAL)
            # 0.55-0.65 = Good match (LIKELY REAL)
            # 0.45-0.55 = Moderate (UNCERTAIN)
            # Below 0.45 = Poor match (LIKELY FAKE)
            
            if max_score >= 0.65 and avg_top3 >= 0.60:
                authenticity = 'REAL'
                confidence = min(98, int(max_score * 100))
                explanation = (
                    f"✅ Strong visual evidence found! "
                    f"Top match similarity: {max_score:.2%}. "
                    f"Multiple matching images confirm this is a documented incident."
                )
                
            elif max_score >= 0.55 and avg_top3 >= 0.50:
                authenticity = 'LIKELY REAL'
                confidence = min(88, int(max_score * 100))
                explanation = (
                    f"✓ Good visual correlation found. "
                    f"Top match: {max_score:.2%}. "
                    f"Evidence suggests this is a genuine incident with online documentation."
                )
                
            elif max_score >= 0.45 or avg_top3 >= 0.42:
                authenticity = 'UNCERTAIN'
                confidence = min(60, int(avg_top3 * 100))
                explanation = (
                    f"⚠️ Moderate correlation detected. "
                    f"Match quality: {max_score:.2%}. "
                    f"Limited visual evidence - requires additional verification."
                )
                
            else:
                authenticity = 'LIKELY FAKE'
                confidence = max(15, int((1 - max_score) * 70))
                explanation = (
                    f"❌ Low visual correlation ({max_score:.2%}). "
                    f"No strong matches found online. "
                    f"This incident may be fabricated, very recent, or misrepresented."
                )
            
            return {
                'authenticity': authenticity,
                'confidence': confidence,
                'explanation': explanation,
                'top_matches': top_matches,
                'avg_similarity': float(avg_top3),
                'max_similarity': float(max_score),
                'all_scores': [s['score'] for s in similarities]
            }
            
        except Exception as e:
            print(f"Verification error: {e}")
            return {
                'authenticity': 'UNCERTAIN',
                'confidence': 0,
                'explanation': f'Verification failed: {str(e)}',
                'top_matches': []
            }
    
    def verify_image_to_text(
        self, 
        query_image: Image.Image,
        retrieved_images: List[Dict]
    ) -> Dict:
        """
        FIXED: Verify uploaded image by comparing with retrieved images
        """
        try:
            # Generate caption for query image
            caption = self.generate_caption(query_image)
            print(f"Generated caption: {caption}")
            
            if not retrieved_images:
                return {
                    'authenticity': 'UNCERTAIN',
                    'confidence': 0,
                    'explanation': 'No similar images found online',
                    'caption': caption,
                    'similar_images': []
                }
            
            similarities = []
            
            # Compare query image with retrieved images using CLIP
            for img_data in retrieved_images:
                try:
                    img = img_data['image']
                    
                    # Use caption for similarity (more accurate than direct image comparison)
                    score = self.compute_clip_similarity(img, caption)
                    
                    similarities.append({
                        'image': img,
                        'score': score,
                        'name': img_data['name'],
                        'url': img_data['url'],
                        'source': img_data['source']
                    })
                    
                    print(f"  • {img_data['name'][:40]}: {score:.4f}")
                    
                except Exception as e:
                    print(f"Error processing image: {e}")
                    continue
            
            if not similarities:
                return {
                    'authenticity': 'UNCERTAIN',
                    'confidence': 0,
                    'explanation': 'Unable to compare images',
                    'caption': caption,
                    'similar_images': []
                }
            
            # Sort by similarity
            similarities.sort(key=lambda x: x['score'], reverse=True)
            similar_images = similarities[:5]
            
            # Calculate statistics
            top_3_scores = [s['score'] for s in similarities[:3]]
            avg_top3 = np.mean(top_3_scores)
            max_score = similarities[0]['score']
            
            print(f"\n📊 Image Match - Max: {max_score:.4f} | Avg(top-3): {avg_top3:.4f}")
            
            # FIXED THRESHOLDS for image verification
            # Slightly higher thresholds for image-image matching
            if max_score >= 0.68 and avg_top3 >= 0.62:
                authenticity = 'REAL'
                confidence = min(97, int(max_score * 100))
                explanation = (
                    f"✅ Matching images found online! "
                    f"Highest similarity: {max_score:.2%}. "
                    f"This incident appears to be documented and verified."
                )
                
            elif max_score >= 0.58 and avg_top3 >= 0.52:
                authenticity = 'LIKELY REAL'
                confidence = min(85, int(max_score * 100))
                explanation = (
                    f"✓ Similar images detected online. "
                    f"Match quality: {max_score:.2%}. "
                    f"Evidence suggests this is a genuine incident."
                )
                
            elif max_score >= 0.48:
                authenticity = 'UNCERTAIN'
                confidence = min(65, int(avg_top3 * 100))
                explanation = (
                    f"⚠️ Partial matches found. "
                    f"Similarity: {max_score:.2%}. "
                    f"Cannot conclusively verify - may require additional sources."
                )
                
            else:
                authenticity = 'LIKELY FAKE'
                confidence = max(15, int((1 - max_score) * 75))
                explanation = (
                    f"❌ No matching images found online ({max_score:.2%}). "
                    f"The image may be fabricated, heavily edited, or not publicly available."
                )
            
            return {
                'authenticity': authenticity,
                'confidence': confidence,
                'explanation': explanation,
                'caption': caption,
                'similar_images': similar_images,
                'avg_similarity': float(avg_top3),
                'max_similarity': float(max_score),
                'all_scores': [s['score'] for s in similarities]
            }
            
        except Exception as e:
            print(f"Image verification error: {e}")
            return {
                'authenticity': 'UNCERTAIN',
                'confidence': 0,
                'explanation': f'Verification failed: {str(e)}',
                'caption': 'Unable to generate caption',
                'similar_images': []
            }
    
    def compute_image_similarity_direct(self, image1: Image.Image, image2: Image.Image) -> float:
        """
        Direct image-to-image similarity using CLIP image encoder
        Optional: More accurate for identical/near-identical images
        """
        try:
            if self.clip_model is None:
                return 0.5
            
            # Preprocess both images
            img1_input = self.clip_preprocess(image1).unsqueeze(0).to(self.device)
            img2_input = self.clip_preprocess(image2).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Get image features
                img1_features = self.clip_model.encode_image(img1_input)
                img2_features = self.clip_model.encode_image(img2_input)
                
                # Normalize
                img1_features = img1_features / img1_features.norm(dim=-1, keepdim=True)
                img2_features = img2_features / img2_features.norm(dim=-1, keepdim=True)
                
                # Compute similarity
                similarity = (img1_features @ img2_features.T).squeeze()
                
                # Normalize to 0-1
                similarity = (similarity + 1) / 2
            
            return float(similarity.cpu())
            
        except Exception as e:
            print(f"Direct image similarity error: {e}")
            return 0.5
