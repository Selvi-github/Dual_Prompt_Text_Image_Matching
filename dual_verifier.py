"""
Dual Verification Module
Handles Text + Image verification with cross-checking
"""

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer, util
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DualVerifier:
    """
    Advanced verification system that handles:
    1. Text + Image together (checks if they match same incident)
    2. Text only (retrieves and verifies images)
    3. Image only (reverse search and verifies)
    """
    
    def __init__(self):
        """Initialize CLIP and other models for cross-modal verification"""
        logger.info("Loading verification models...")
        
        # CLIP for text-image matching
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Sentence transformer for text similarity
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        logger.info("✓ Models loaded successfully")
    
    def verify_text_and_image(
        self, 
        user_text: str, 
        user_image: Image.Image,
        text_retrieved_images: List[Dict],
        image_retrieved_images: List[Dict]
    ) -> Dict:
        """
        Main verification function for Text + Image input
        
        Args:
            user_text: User provided incident description
            user_image: User provided incident image
            text_retrieved_images: Images retrieved based on text search
            image_retrieved_images: Images retrieved based on reverse image search
            
        Returns:
            Comprehensive verification result
        """
        logger.info("🔍 Starting dual verification (Text + Image)...")
        
        # Step 1: Check if user image matches user text
        text_image_similarity = self._calculate_clip_similarity(user_text, user_image)
        logger.info(f"Text-Image similarity: {text_image_similarity:.4f}")
        
        # Step 2: Verify text against web images
        text_verification = self._verify_against_web_images(
            user_text, 
            text_retrieved_images,
            mode='text'
        )
        
        # Step 3: Verify image against web images
        image_verification = self._verify_against_web_images(
            user_image,
            image_retrieved_images,
            mode='image'
        )
        
        # Step 4: Cross-check if both refer to same incident
        same_incident = self._check_same_incident(
            text_retrieved_images,
            image_retrieved_images
        )
        
        # Step 5: Generate final verdict
        result = self._generate_dual_verdict(
            text_image_similarity,
            text_verification,
            image_verification,
            same_incident
        )
        
        return result
    
    def verify_text_only(self, user_text: str, retrieved_images: List[Dict]) -> Dict:
        """
        Verification when only text is provided
        """
        logger.info("🔍 Starting text-only verification...")
        
        verification = self._verify_against_web_images(
            user_text,
            retrieved_images,
            mode='text'
        )
        
        return {
            'mode': 'text_only',
            'authenticity': verification['authenticity'],
            'confidence': verification['confidence'],
            'explanation': verification['explanation'],
            'top_matches': verification['top_matches'],
            'is_real': verification['is_real']
        }
    
    def verify_image_only(self, user_image: Image.Image, retrieved_images: List[Dict]) -> Dict:
        """
        Verification when only image is provided
        """
        logger.info("🔍 Starting image-only verification...")
        
        verification = self._verify_against_web_images(
            user_image,
            retrieved_images,
            mode='image'
        )
        
        return {
            'mode': 'image_only',
            'authenticity': verification['authenticity'],
            'confidence': verification['confidence'],
            'explanation': verification['explanation'],
            'top_matches': verification['top_matches'],
            'is_real': verification['is_real']
        }
    
    def _calculate_clip_similarity(self, text: str, image: Image.Image) -> float:
        """
        Calculate similarity between text and image using CLIP
        """
        try:
            inputs = self.clip_processor(
                text=[text],
                images=image,
                return_tensors="pt",
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                
            # Get similarity score
            logits_per_image = outputs.logits_per_image
            similarity = torch.softmax(logits_per_image, dim=1)[0][0].item()
            
            return similarity
            
        except Exception as e:
            logger.error(f"CLIP similarity calculation failed: {e}")
            return 0.5
    
    def _verify_against_web_images(
        self,
        query,  # Can be text (str) or image (PIL.Image)
        retrieved_images: List[Dict],
        mode: str  # 'text' or 'image'
    ) -> Dict:
        """
        Verify query against retrieved web images
        """
        if not retrieved_images:
            return {
                'authenticity': 'UNCERTAIN',
                'confidence': 0,
                'explanation': 'No images found for comparison',
                'top_matches': [],
                'is_real': False
            }
        
        similarities = []
        
        for img_data in retrieved_images:
            if mode == 'text':
                # Text vs Image comparison
                similarity = self._calculate_clip_similarity(query, img_data['image'])
            else:
                # Image vs Image comparison
                similarity = self._calculate_image_similarity(query, img_data['image'])
            
            similarities.append({
                'score': similarity,
                'image': img_data['image'],
                'source': img_data['source'],
                'name': img_data['name'],
                'url': img_data.get('url', '')
            })
        
        # Sort by similarity
        similarities.sort(key=lambda x: x['score'], reverse=True)
        top_matches = similarities[:5]
        
        # Calculate average similarity
        avg_similarity = np.mean([s['score'] for s in similarities])
        max_similarity = similarities[0]['score'] if similarities else 0
        
        # Determine authenticity
        if max_similarity > 0.7 and avg_similarity > 0.5:
            authenticity = 'REAL'
            confidence = int(min(95, max_similarity * 100))
            explanation = f"Strong match found with credible sources. Multiple similar images detected online."
            is_real = True
        elif max_similarity > 0.5 and avg_similarity > 0.35:
            authenticity = 'LIKELY REAL'
            confidence = int(max_similarity * 100)
            explanation = f"Moderate match found. Some visual correlation exists with online sources."
            is_real = True
        elif max_similarity > 0.3:
            authenticity = 'UNCERTAIN'
            confidence = int(max_similarity * 100)
            explanation = f"Weak correlation found. Limited evidence available online."
            is_real = False
        else:
            authenticity = 'LIKELY FAKE'
            confidence = int((1 - max_similarity) * 100)
            explanation = f"No matching images found online. This may be fabricated or very recent."
            is_real = False
        
        return {
            'authenticity': authenticity,
            'confidence': confidence,
            'explanation': explanation,
            'top_matches': top_matches,
            'is_real': is_real,
            'avg_similarity': avg_similarity,
            'max_similarity': max_similarity
        }
    
    def _calculate_image_similarity(self, image1: Image.Image, image2: Image.Image) -> float:
        """
        Calculate similarity between two images using CLIP embeddings
        """
        try:
            # Process both images
            inputs1 = self.clip_processor(images=image1, return_tensors="pt")
            inputs2 = self.clip_processor(images=image2, return_tensors="pt")
            
            with torch.no_grad():
                embedding1 = self.clip_model.get_image_features(**inputs1)
                embedding2 = self.clip_model.get_image_features(**inputs2)
            
            # Normalize embeddings
            embedding1 = embedding1 / embedding1.norm(dim=-1, keepdim=True)
            embedding2 = embedding2 / embedding2.norm(dim=-1, keepdim=True)
            
            # Calculate cosine similarity
            similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2)
            
            return similarity.item()
            
        except Exception as e:
            logger.error(f"Image similarity calculation failed: {e}")
            return 0.3
    
    def _check_same_incident(
        self,
        text_images: List[Dict],
        image_images: List[Dict]
    ) -> Dict:
        """
        Check if text-based and image-based searches refer to same incident
        """
        if not text_images or not image_images:
            return {
                'same_incident': False,
                'confidence': 0,
                'cross_match_score': 0
            }
        
        # Compare top images from both searches
        cross_similarities = []
        
        for text_img in text_images[:3]:
            for img_img in image_images[:3]:
                sim = self._calculate_image_similarity(
                    text_img['image'],
                    img_img['image']
                )
                cross_similarities.append(sim)
        
        if cross_similarities:
            max_cross_sim = max(cross_similarities)
            avg_cross_sim = np.mean(cross_similarities)
            
            # If images from both searches match each other → same incident
            if max_cross_sim > 0.7:
                return {
                    'same_incident': True,
                    'confidence': int(max_cross_sim * 100),
                    'cross_match_score': max_cross_sim
                }
            elif max_cross_sim > 0.4:
                return {
                    'same_incident': 'UNCERTAIN',
                    'confidence': int(max_cross_sim * 100),
                    'cross_match_score': max_cross_sim
                }
        
        return {
            'same_incident': False,
            'confidence': 0,
            'cross_match_score': 0
        }
    
    def _generate_dual_verdict(
        self,
        text_image_similarity: float,
        text_verification: Dict,
        image_verification: Dict,
        same_incident: Dict
    ) -> Dict:
        """
        Generate final verdict combining all checks
        """
        logger.info("📊 Generating final verdict...")
        
        # Scenario 1: Text and Image MATCH + BOTH REAL + SAME INCIDENT
        if (text_image_similarity > 0.6 and 
            text_verification['is_real'] and 
            image_verification['is_real'] and
            same_incident['same_incident'] == True):
            
            return {
                'verdict': 'MATCH_AND_REAL',
                'authenticity': 'REAL',
                'confidence': min(95, int((text_image_similarity + 
                                          text_verification['max_similarity'] + 
                                          image_verification['max_similarity']) / 3 * 100)),
                'main_message': '✅ TEXT and IMAGE are from the SAME REAL INCIDENT',
                'explanation': (
                    f"High confidence verification:\n"
                    f"• Text-Image match: {text_image_similarity:.2%}\n"
                    f"• Text verified against {len(text_verification['top_matches'])} web sources\n"
                    f"• Image verified against {len(image_verification['top_matches'])} web sources\n"
                    f"• Cross-verification confirms same incident (confidence: {same_incident['confidence']}%)\n\n"
                    f"This incident is REAL and both text and image describe the same event."
                ),
                'text_verification': text_verification,
                'image_verification': image_verification,
                'text_image_similarity': text_image_similarity,
                'same_incident': True
            }
        
        # Scenario 2: BOTH REAL but DIFFERENT INCIDENTS
        elif (text_verification['is_real'] and 
              image_verification['is_real'] and
              same_incident['same_incident'] == False):
            
            return {
                'verdict': 'BOTH_REAL_DIFFERENT_INCIDENTS',
                'authenticity': 'MISMATCH',
                'confidence': 75,
                'main_message': '⚠️ BOTH are REAL incidents, but they are NOT THE SAME INCIDENT',
                'explanation': (
                    f"Verification results:\n\n"
                    f"📝 TEXT describes a REAL incident:\n"
                    f"   • Confidence: {text_verification['confidence']}%\n"
                    f"   • {text_verification['explanation']}\n\n"
                    f"🖼️ IMAGE shows a DIFFERENT REAL incident:\n"
                    f"   • Confidence: {image_verification['confidence']}%\n"
                    f"   • {image_verification['explanation']}\n\n"
                    f"⚠️ WARNING: Text and image are from DIFFERENT incidents!\n"
                    f"This could be:\n"
                    f"• Accidental mismatch\n"
                    f"• Deliberate misinformation\n"
                    f"• Context confusion\n\n"
                    f"Both incidents are real, but they are NOT related."
                ),
                'text_verification': text_verification,
                'image_verification': image_verification,
                'text_image_similarity': text_image_similarity,
                'same_incident': False
            }
        
        # Scenario 3: Text REAL but Image FAKE (or vice versa)
        elif text_verification['is_real'] != image_verification['is_real']:
            
            real_part = 'TEXT' if text_verification['is_real'] else 'IMAGE'
            fake_part = 'IMAGE' if text_verification['is_real'] else 'TEXT'
            
            return {
                'verdict': 'PARTIAL_FAKE',
                'authenticity': 'LIKELY FAKE',
                'confidence': 60,
                'main_message': f'❌ {real_part} is REAL but {fake_part} is LIKELY FAKE',
                'explanation': (
                    f"Mixed verification results:\n\n"
                    f"📝 TEXT verification:\n"
                    f"   • Status: {'REAL' if text_verification['is_real'] else 'FAKE'}\n"
                    f"   • Confidence: {text_verification['confidence']}%\n"
                    f"   • {text_verification['explanation']}\n\n"
                    f"🖼️ IMAGE verification:\n"
                    f"   • Status: {'REAL' if image_verification['is_real'] else 'FAKE'}\n"
                    f"   • Confidence: {image_verification['confidence']}%\n"
                    f"   • {image_verification['explanation']}\n\n"
                    f"⚠️ This is likely MISINFORMATION - combining real and fake content."
                ),
                'text_verification': text_verification,
                'image_verification': image_verification,
                'text_image_similarity': text_image_similarity,
                'same_incident': False
            }
        
        # Scenario 4: Low similarity between text and image
        elif text_image_similarity < 0.4:
            
            return {
                'verdict': 'LOW_MATCH',
                'authenticity': 'UNCERTAIN',
                'confidence': int(text_image_similarity * 100),
                'main_message': '⚠️ TEXT and IMAGE do not match well',
                'explanation': (
                    f"Poor text-image correlation (similarity: {text_image_similarity:.2%})\n\n"
                    f"The provided text and image may not be describing the same thing.\n"
                    f"Please verify that:\n"
                    f"• Image matches the incident description\n"
                    f"• Text accurately describes the image\n"
                    f"• Both are from the same incident\n\n"
                    f"Individual verifications:\n"
                    f"• Text: {text_verification['authenticity']}\n"
                    f"• Image: {image_verification['authenticity']}"
                ),
                'text_verification': text_verification,
                'image_verification': image_verification,
                'text_image_similarity': text_image_similarity,
                'same_incident': same_incident['same_incident']
            }
        
        # Scenario 5: Both appear fake or uncertain
        else:
            return {
                'verdict': 'UNCERTAIN_OR_FAKE',
                'authenticity': 'UNCERTAIN',
                'confidence': 40,
                'main_message': '❓ Unable to verify authenticity',
                'explanation': (
                    f"Limited evidence found:\n\n"
                    f"📝 TEXT verification:\n"
                    f"   • {text_verification['authenticity']}\n"
                    f"   • Confidence: {text_verification['confidence']}%\n\n"
                    f"🖼️ IMAGE verification:\n"
                    f"   • {image_verification['authenticity']}\n"
                    f"   • Confidence: {image_verification['confidence']}%\n\n"
                    f"This incident may be:\n"
                    f"• Very recent (not yet indexed online)\n"
                    f"• Fabricated/fake\n"
                    f"• From limited coverage sources\n\n"
                    f"Further investigation recommended."
                ),
                'text_verification': text_verification,
                'image_verification': image_verification,
                'text_image_similarity': text_image_similarity,
                'same_incident': same_incident['same_incident']
            }


# Example usage
if __name__ == "__main__":
    print("🔍 Dual Verification System - Ready!")
    print("\nSupports:")
    print("1. Text + Image verification")
    print("2. Text only verification")
    print("3. Image only verification")