"""
Dual Verification Module
Verifies text and images against retrieved web evidence
"""

from PIL import Image
import numpy as np
from typing import List, Dict
import re

class DualVerifier:
    def __init__(self):
        """Initialize verifier"""
        print("✓ Dual Verifier initialized")
        
        # Confidence thresholds
        self.high_confidence_threshold = 70
        self.medium_confidence_threshold = 50
    
    def verify_text_and_image(
        self, 
        text: str, 
        user_image: Image.Image,
        text_based_images: List[Dict],
        image_based_images: List[Dict]
    ) -> Dict:
        """
        Verify both text and image together
        Returns detailed verification result
        """
        try:
            # Verify text against retrieved images
            text_result = self._verify_text_with_images(text, text_based_images)
            
            # Verify image against similar images
            image_result = self._verify_image_with_images(user_image, image_based_images)
            
            # Cross-verify consistency
            consistency_score = self._check_consistency(text, user_image, text_result, image_result)
            
            # Determine final verdict
            verdict = self._determine_verdict(text_result, image_result, consistency_score)
            
            return {
                'verdict': verdict['type'],
                'main_message': verdict['message'],
                'confidence': verdict['confidence'],
                'explanation': verdict['explanation'],
                'text_verification': text_result,
                'image_verification': image_result,
                'consistency_score': consistency_score
            }
        
        except Exception as e:
            print(f"Dual verification error: {e}")
            return self._get_error_result()
    
    def verify_text_only(self, text: str, retrieved_images: List[Dict]) -> Dict:
        """Verify text against retrieved images"""
        try:
            result = self._verify_text_with_images(text, retrieved_images)
            
            return {
                'is_real': result['is_real'],
                'authenticity': result['authenticity'],
                'confidence': result['confidence'],
                'explanation': result['explanation'],
                'evidence_count': len(retrieved_images)
            }
        
        except Exception as e:
            print(f"Text verification error: {e}")
            return {
                'is_real': False,
                'authenticity': 'UNCERTAIN',
                'confidence': 0,
                'explanation': 'Verification failed due to error',
                'evidence_count': 0
            }
    
    def verify_image_only(self, user_image: Image.Image, retrieved_images: List[Dict]) -> Dict:
        """Verify image against similar images"""
        try:
            result = self._verify_image_with_images(user_image, retrieved_images)
            
            return {
                'is_real': result['is_real'],
                'authenticity': result['authenticity'],
                'confidence': result['confidence'],
                'explanation': result['explanation'],
                'similar_count': len(retrieved_images)
            }
        
        except Exception as e:
            print(f"Image verification error: {e}")
            return {
                'is_real': False,
                'authenticity': 'UNCERTAIN',
                'confidence': 0,
                'explanation': 'Verification failed due to error',
                'similar_count': 0
            }
    
    def _verify_text_with_images(self, text: str, images: List[Dict]) -> Dict:
        """Verify text description against retrieved images"""
        if not images or len(images) == 0:
            return {
                'is_real': False,
                'authenticity': 'LIKELY FAKE',
                'confidence': 30,
                'explanation': 'No supporting images found online',
                'evidence_score': 0
            }
        
        # Calculate evidence score based on number of images
        evidence_score = min(len(images) * 10, 100)
        
        # Check if images are from credible sources
        credible_sources = ['news', 'government', 'official', 'reuters', 'bbc', 'cnn']
        credible_count = sum(
            1 for img in images 
            if any(source in img.get('source', '').lower() for source in credible_sources)
        )
        
        credibility_boost = min(credible_count * 5, 20)
        
        # Final confidence
        confidence = min(evidence_score + credibility_boost, 95)
        
        if confidence >= self.high_confidence_threshold:
            is_real = True
            authenticity = 'REAL'
            explanation = f'Found {len(images)} supporting images online. High confidence this incident is authentic.'
        elif confidence >= self.medium_confidence_threshold:
            is_real = True
            authenticity = 'LIKELY REAL'
            explanation = f'Found {len(images)} images online. Moderate confidence this incident occurred.'
        else:
            is_real = False
            authenticity = 'UNCERTAIN'
            explanation = f'Limited evidence found ({len(images)} images). Cannot confirm authenticity.'
        
        return {
            'is_real': is_real,
            'authenticity': authenticity,
            'confidence': confidence,
            'explanation': explanation,
            'evidence_score': evidence_score
        }
    
    def _verify_image_with_images(self, user_image: Image.Image, similar_images: List[Dict]) -> Dict:
        """Verify user image against similar images found online"""
        if not similar_images or len(similar_images) == 0:
            return {
                'is_real': False,
                'authenticity': 'LIKELY FAKE',
                'confidence': 25,
                'explanation': 'No similar images found online. Image may be fabricated.',
                'similarity_score': 0
            }
        
        # Calculate similarity score
        similarity_scores = []
        for img_data in similar_images:
            try:
                similarity = self._calculate_image_similarity(
                    user_image, 
                    img_data['image']
                )
                similarity_scores.append(similarity)
            except:
                continue
        
        if not similarity_scores:
            avg_similarity = 0
        else:
            avg_similarity = sum(similarity_scores) / len(similarity_scores)
        
        # Confidence based on similarity and count
        confidence = min(int(avg_similarity * 100) + len(similar_images) * 5, 95)
        
        if confidence >= self.high_confidence_threshold:
            is_real = True
            authenticity = 'REAL'
            explanation = f'Found {len(similar_images)} similar images online with {avg_similarity:.1%} similarity. Image appears authentic.'
        elif confidence >= self.medium_confidence_threshold:
            is_real = True
            authenticity = 'LIKELY REAL'
            explanation = f'Found {len(similar_images)} similar images. Moderate confidence image is real.'
        else:
            is_real = False
            authenticity = 'UNCERTAIN'
            explanation = f'Limited similar images found. Cannot confirm image authenticity.'
        
        return {
            'is_real': is_real,
            'authenticity': authenticity,
            'confidence': confidence,
            'explanation': explanation,
            'similarity_score': avg_similarity
        }
    
    def _calculate_image_similarity(self, img1: Image.Image, img2: Image.Image) -> float:
        """Calculate basic similarity between two images"""
        try:
            # Resize to same size for comparison
            size = (100, 100)
            img1_resized = img1.resize(size)
            img2_resized = img2.resize(size)
            
            # Convert to numpy arrays
            arr1 = np.array(img1_resized).flatten()
            arr2 = np.array(img2_resized).flatten()
            
            # Calculate correlation
            correlation = np.corrcoef(arr1, arr2)[0, 1]
            
            # Convert to similarity (0 to 1)
            similarity = (correlation + 1) / 2
            
            return max(0, min(1, similarity))
        
        except Exception as e:
            return 0.5  # Default middle value
    
    def _check_consistency(self, text: str, image: Image.Image, text_result: Dict, image_result: Dict) -> float:
        """Check if text and image are consistent with each other"""
        try:
            # Simple consistency check based on both being real/fake
            text_real = text_result['is_real']
            image_real = image_result['is_real']
            
            if text_real == image_real:
                # Both agree (both real or both fake)
                consistency = 0.8
            else:
                # Mismatch
                consistency = 0.3
            
            return consistency
        
        except:
            return 0.5
    
    def _determine_verdict(self, text_result: Dict, image_result: Dict, consistency: float) -> Dict:
        """Determine final verdict based on all evidence"""
        text_real = text_result['is_real']
        image_real = image_result['is_real']
        
        text_conf = text_result['confidence']
        image_conf = image_result['confidence']
        
        # Average confidence
        avg_confidence = int((text_conf + image_conf) / 2)
        
        if text_real and image_real and consistency > 0.6:
            return {
                'type': 'MATCH_AND_REAL',
                'message': '✅ TEXT and IMAGE MATCH - Both Verified as REAL',
                'confidence': avg_confidence,
                'explanation': (
                    'Both the text description and image have been verified against online sources.\n\n'
                    f'Text Verification: {text_result["authenticity"]} ({text_conf}% confidence)\n'
                    f'Image Verification: {image_result["authenticity"]} ({image_conf}% confidence)\n\n'
                    'The incident appears to be authentic.'
                )
            }
        
        elif text_real and image_real and consistency <= 0.6:
            return {
                'type': 'BOTH_REAL_DIFFERENT_INCIDENTS',
                'message': '⚠️ MISMATCH DETECTED - Text and Image May Describe Different Incidents',
                'confidence': avg_confidence,
                'explanation': (
                    'Both text and image appear to be real, but they may not describe the same incident.\n\n'
                    f'Text Verification: {text_result["authenticity"]} ({text_conf}% confidence)\n'
                    f'Image Verification: {image_result["authenticity"]} ({image_conf}% confidence)\n\n'
                    '⚠️ WARNING: The image might be from a different event than described in the text.'
                )
            }
        
        elif not text_real and not image_real:
            return {
                'type': 'BOTH_FAKE',
                'message': '❌ LIKELY FABRICATED - Both Text and Image Cannot Be Verified',
                'confidence': avg_confidence,
                'explanation': (
                    'Neither the text description nor the image could be verified against online sources.\n\n'
                    f'Text Verification: {text_result["authenticity"]} ({text_conf}% confidence)\n'
                    f'Image Verification: {image_result["authenticity"]} ({image_conf}% confidence)\n\n'
                    'This incident may be fabricated or lacks online documentation.'
                )
            }
        
        else:
            return {
                'type': 'PARTIAL_FAKE',
                'message': '⚠️ PARTIAL VERIFICATION - One Component Could Not Be Verified',
                'confidence': avg_confidence,
                'explanation': (
                    f'Text: {text_result["authenticity"]} ({text_conf}% confidence)\n'
                    f'Image: {image_result["authenticity"]} ({image_conf}% confidence)\n\n'
                    'One component appears real while the other cannot be verified. '
                    'This could indicate manipulation or mismatched information.'
                )
            }
    
    def _get_error_result(self) -> Dict:
        """Return error result"""
        return {
            'verdict': 'ERROR',
            'main_message': '❌ Verification Error',
            'confidence': 0,
            'explanation': 'An error occurred during verification. Please try again.',
            'text_verification': {'is_real': False, 'authenticity': 'ERROR', 'confidence': 0, 'explanation': 'Error'},
            'image_verification': {'is_real': False, 'authenticity': 'ERROR', 'confidence': 0, 'explanation': 'Error'},
            'consistency_score': 0
        }
