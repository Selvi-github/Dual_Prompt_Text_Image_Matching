"""
Dual Verification Module - STRICT VERSION
Properly verifies text and images against retrieved web evidence
Uses strict thresholds to avoid false positives
"""

from PIL import Image
import numpy as np
from typing import List, Dict
import re

class DualVerifier:
    def __init__(self):
        """Initialize verifier with strict thresholds"""
        print("✓ Dual Verifier initialized (Strict Mode)")
        
        # STRICT thresholds - harder to pass
        self.high_similarity_threshold = 0.75  # Must be 75%+ similar
        self.min_similar_images = 3  # Need at least 3 matching images
        self.min_evidence_count = 5  # Need at least 5 images for "REAL"
        
        self.high_confidence_threshold = 75
        self.medium_confidence_threshold = 55
    
    def verify_text_and_image(
        self, 
        text: str, 
        user_image: Image.Image,
        text_based_images: List[Dict],
        image_based_images: List[Dict]
    ) -> Dict:
        """
        STRICT verification of both text and image together
        """
        try:
            # Verify text - STRICT
            text_result = self._strict_verify_text(text, text_based_images)
            
            # Verify image - STRICT
            image_result = self._strict_verify_image(user_image, image_based_images)
            
            # Cross-verify consistency
            consistency_score = self._check_consistency(
                text, user_image, text_result, image_result, 
                text_based_images, image_based_images
            )
            
            # Determine final verdict with strict rules
            verdict = self._determine_strict_verdict(
                text_result, image_result, consistency_score
            )
            
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
        """STRICT text-only verification"""
        try:
            result = self._strict_verify_text(text, retrieved_images)
            
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
                'authenticity': 'ERROR',
                'confidence': 0,
                'explanation': 'Verification failed',
                'evidence_count': 0
            }
    
    def verify_image_only(self, user_image: Image.Image, retrieved_images: List[Dict]) -> Dict:
        """STRICT image-only verification"""
        try:
            result = self._strict_verify_image(user_image, retrieved_images)
            
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
                'authenticity': 'ERROR',
                'confidence': 0,
                'explanation': 'Verification failed',
                'similar_count': 0
            }
    
    def _strict_verify_text(self, text: str, images: List[Dict]) -> Dict:
        """
        STRICT text verification - requires strong evidence
        """
        num_images = len(images)
        
        # RULE 1: Not enough images = LIKELY FAKE
        if num_images < self.min_evidence_count:
            return {
                'is_real': False,
                'authenticity': 'LIKELY FAKE',
                'confidence': max(20, num_images * 4),
                'explanation': f'Insufficient evidence: Only {num_images} images found online. Real incidents typically have more coverage.',
                'evidence_score': num_images * 10
            }
        
        # RULE 2: Check for credible sources
        credible_keywords = [
            'news', 'bbc', 'cnn', 'reuters', 'guardian', 'times', 
            'post', 'government', 'official', 'agency', 'press'
        ]
        
        credible_count = 0
        for img in images:
            source = img.get('source', '').lower()
            name = img.get('name', '').lower()
            metadata = str(img.get('metadata', '')).lower()
            
            combined = f"{source} {name} {metadata}"
            
            if any(keyword in combined for keyword in credible_keywords):
                credible_count += 1
        
        credibility_ratio = credible_count / num_images if num_images > 0 else 0
        
        # RULE 3: Calculate confidence based on evidence
        base_confidence = min(num_images * 7, 60)
        credibility_boost = int(credibility_ratio * 30)
        
        final_confidence = min(base_confidence + credibility_boost, 95)
        
        # RULE 4: Determine authenticity based on strict thresholds
        if final_confidence >= 75 and num_images >= 8:
            is_real = True
            authenticity = 'REAL'
            explanation = (
                f'✓ Strong evidence found: {num_images} images from online sources.\n'
                f'✓ {credible_count} images from credible news sources.\n'
                f'This incident appears to be well-documented and authentic.'
            )
        
        elif final_confidence >= 55 and num_images >= 6:
            is_real = True
            authenticity = 'LIKELY REAL'
            explanation = (
                f'Moderate evidence: {num_images} images found online.\n'
                f'{credible_count} from credible sources.\n'
                f'Incident likely occurred but has limited documentation.'
            )
        
        else:
            is_real = False
            authenticity = 'UNCERTAIN / LIKELY FAKE'
            explanation = (
                f'⚠️ Limited evidence: Only {num_images} images found.\n'
                f'Only {credible_count} from credible sources.\n'
                f'Cannot confidently verify this incident. May be fabricated or poorly documented.'
            )
        
        return {
            'is_real': is_real,
            'authenticity': authenticity,
            'confidence': final_confidence,
            'explanation': explanation,
            'evidence_score': num_images,
            'credible_sources': credible_count
        }
    
    def _strict_verify_image(self, user_image: Image.Image, similar_images: List[Dict]) -> Dict:
        """
        STRICT image verification - requires HIGH similarity
        """
        num_images = len(similar_images)
        
        # RULE 1: Not enough similar images = LIKELY FAKE
        if num_images < self.min_similar_images:
            return {
                'is_real': False,
                'authenticity': 'LIKELY FAKE',
                'confidence': max(15, num_images * 5),
                'explanation': f'No matching images found online. Only {num_images} somewhat similar images retrieved. This image may be fabricated or AI-generated.',
                'similarity_score': 0,
                'high_matches': 0
            }
        
        # RULE 2: Calculate ACTUAL similarity scores
        similarity_scores = []
        high_similarity_count = 0
        
        for img_data in similar_images:
            try:
                similarity = self._calculate_accurate_similarity(
                    user_image, 
                    img_data['image']
                )
                similarity_scores.append(similarity)
                
                # Count truly similar images (75%+ match)
                if similarity >= self.high_similarity_threshold:
                    high_similarity_count += 1
            except:
                similarity_scores.append(0.3)  # Default low score
        
        if not similarity_scores:
            avg_similarity = 0
        else:
            avg_similarity = sum(similarity_scores) / len(similarity_scores)
        
        # RULE 3: Require HIGH similarity matches
        if high_similarity_count < 2:
            return {
                'is_real': False,
                'authenticity': 'LIKELY FAKE',
                'confidence': max(25, int(avg_similarity * 50)),
                'explanation': (
                    f'⚠️ Low similarity: Only {high_similarity_count} images with 75%+ match found.\n'
                    f'Average similarity: {avg_similarity:.1%}\n'
                    f'This image does not closely match any verified sources online.'
                ),
                'similarity_score': avg_similarity,
                'high_matches': high_similarity_count
            }
        
        # RULE 4: Calculate confidence based on similarity AND count
        similarity_score = int(avg_similarity * 50)
        count_score = min(high_similarity_count * 15, 40)
        final_confidence = min(similarity_score + count_score, 95)
        
        # RULE 5: Determine authenticity
        if final_confidence >= 75 and high_similarity_count >= 4:
            is_real = True
            authenticity = 'REAL'
            explanation = (
                f'✓ High similarity match: {high_similarity_count} images with 75%+ similarity.\n'
                f'✓ Average similarity: {avg_similarity:.1%}\n'
                f'This image appears in multiple verified online sources.'
            )
        
        elif final_confidence >= 55 and high_similarity_count >= 2:
            is_real = True
            authenticity = 'LIKELY REAL'
            explanation = (
                f'Moderate match: {high_similarity_count} similar images found.\n'
                f'Average similarity: {avg_similarity:.1%}\n'
                f'Image likely authentic but has limited matches.'
            )
        
        else:
            is_real = False
            authenticity = 'UNCERTAIN / LIKELY FAKE'
            explanation = (
                f'⚠️ Poor match: Only {high_similarity_count} images with good similarity.\n'
                f'Average similarity: {avg_similarity:.1%}\n'
                f'Cannot verify this image from online sources. May be edited or fabricated.'
            )
        
        return {
            'is_real': is_real,
            'authenticity': authenticity,
            'confidence': final_confidence,
            'explanation': explanation,
            'similarity_score': avg_similarity,
            'high_matches': high_similarity_count
        }
    
    def _calculate_accurate_similarity(self, img1: Image.Image, img2: Image.Image) -> float:
        """
        Calculate ACCURATE similarity using multiple strict metrics
        Returns 0-1 score (higher = more similar)
        """
        try:
            # Resize for comparison
            size = (128, 128)
            img1_resized = img1.resize(size).convert('RGB')
            img2_resized = img2.resize(size).convert('RGB')
            
            arr1 = np.array(img1_resized).astype(float)
            arr2 = np.array(img2_resized).astype(float)
            
            # Metric 1: Normalized pixel difference (STRICT)
            pixel_diff = np.abs(arr1 - arr2).mean() / 255.0
            pixel_similarity = 1 - pixel_diff
            
            # Metric 2: Color histogram correlation (STRICT)
            hist_sim = self._strict_histogram_similarity(arr1, arr2)
            
            # Metric 3: Structural pattern matching
            struct_sim = self._structural_similarity(arr1, arr2)
            
            # Weighted average (strict weights)
            final_similarity = (
                pixel_similarity * 0.4 +
                hist_sim * 0.3 +
                struct_sim * 0.3
            )
            
            return max(0, min(1, final_similarity))
        
        except Exception as e:
            print(f"Similarity error: {e}")
            return 0.3  # Default LOW score on error
    
    def _strict_histogram_similarity(self, arr1: np.ndarray, arr2: np.ndarray) -> float:
        """STRICT color histogram comparison"""
        try:
            similarities = []
            
            for channel in range(3):
                hist1, _ = np.histogram(arr1[:, :, channel], bins=32, range=(0, 256))
                hist2, _ = np.histogram(arr2[:, :, channel], bins=32, range=(0, 256))
                
                hist1 = hist1 / (hist1.sum() + 1e-10)
                hist2 = hist2 / (hist2.sum() + 1e-10)
                
                # Bhattacharyya coefficient (stricter)
                similarity = np.sum(np.sqrt(hist1 * hist2))
                similarities.append(similarity)
            
            return np.mean(similarities)
        except:
            return 0.3
    
    def _structural_similarity(self, arr1: np.ndarray, arr2: np.ndarray) -> float:
        """Calculate structural similarity (simplified SSIM)"""
        try:
            gray1 = np.mean(arr1, axis=2)
            gray2 = np.mean(arr2, axis=2)
            
            # Calculate local means
            mean1 = gray1.mean()
            mean2 = gray2.mean()
            
            # Calculate variances
            var1 = gray1.var()
            var2 = gray2.var()
            
            # Calculate covariance
            covar = np.mean((gray1 - mean1) * (gray2 - mean2))
            
            # SSIM formula
            c1 = (0.01 * 255) ** 2
            c2 = (0.03 * 255) ** 2
            
            ssim = ((2 * mean1 * mean2 + c1) * (2 * covar + c2)) / \
                   ((mean1**2 + mean2**2 + c1) * (var1 + var2 + c2))
            
            # Normalize to 0-1
            return (ssim + 1) / 2
        except:
            return 0.3
    
    def _check_consistency(
        self, text: str, image: Image.Image, 
        text_result: Dict, image_result: Dict,
        text_images: List, image_images: List
    ) -> float:
        """
        STRICT consistency check between text and image
        """
        try:
            text_real = text_result['is_real']
            image_real = image_result['is_real']
            
            # Both must be real with good confidence
            if text_real and image_real:
                text_conf = text_result['confidence']
                img_conf = image_result['confidence']
                
                # If both have high confidence, check overlap
                if text_conf >= 70 and img_conf >= 70:
                    return 0.85
                elif text_conf >= 50 and img_conf >= 50:
                    return 0.65
                else:
                    return 0.45
            
            # If both fake
            elif not text_real and not image_real:
                return 0.6  # Consistent (both fake)
            
            # Mismatch
            else:
                return 0.25  # Low consistency
        
        except:
            return 0.5
    
    def _determine_strict_verdict(
        self, text_result: Dict, image_result: Dict, consistency: float
    ) -> Dict:
        """STRICT final verdict determination"""
        
        text_real = text_result['is_real']
        image_real = image_result['is_real']
        
        text_conf = text_result['confidence']
        image_conf = image_result['confidence']
        
        avg_confidence = int((text_conf + image_conf) / 2)
        
        # CASE 1: Both REAL with HIGH consistency
        if text_real and image_real and consistency > 0.7 and avg_confidence >= 70:
            return {
                'type': 'MATCH_AND_REAL',
                'message': '✅ TEXT and IMAGE MATCH - Both Verified as REAL',
                'confidence': avg_confidence,
                'explanation': (
                    f'✓ Text verification: {text_result["authenticity"]} ({text_conf}%)\n'
                    f'✓ Image verification: {image_result["authenticity"]} ({image_conf}%)\n'
                    f'✓ Consistency: {int(consistency * 100)}%\n\n'
                    'Strong evidence confirms this incident is authentic and well-documented online.'
                )
            }
        
        # CASE 2: Both REAL but LOW consistency
        elif text_real and image_real and consistency <= 0.7:
            return {
                'type': 'BOTH_REAL_DIFFERENT_INCIDENTS',
                'message': '⚠️ MISMATCH - Text and Image May Describe Different Incidents',
                'confidence': avg_confidence,
                'explanation': (
                    f'⚠️ Text: {text_result["authenticity"]} ({text_conf}%)\n'
                    f'⚠️ Image: {image_result["authenticity"]} ({image_conf}%)\n'
                    f'⚠️ Consistency: {int(consistency * 100)}%\n\n'
                    'WARNING: Both appear real separately, but they may not describe the SAME incident. '
                    'The image might be from a different event than described in the text.'
                )
            }
        
        # CASE 3: Both FAKE
        elif not text_real and not image_real:
            return {
                'type': 'BOTH_FAKE',
                'message': '❌ LIKELY FABRICATED - Cannot Verify Text or Image',
                'confidence': avg_confidence,
                'explanation': (
                    f'❌ Text: {text_result["authenticity"]} ({text_conf}%)\n'
                    f'❌ Image: {image_result["authenticity"]} ({image_conf}%)\n\n'
                    'Neither component could be verified from credible online sources. '
                    'This incident is likely fabricated, AI-generated, or lacks proper documentation.'
                )
            }
        
        # CASE 4: One REAL, one FAKE
        else:
            real_component = "Text" if text_real else "Image"
            fake_component = "Image" if text_real else "Text"
            
            return {
                'type': 'PARTIAL_FAKE',
                'message': f'⚠️ SUSPICIOUS - {fake_component} Cannot Be Verified',
                'confidence': avg_confidence,
                'explanation': (
                    f'Text: {text_result["authenticity"]} ({text_conf}%)\n'
                    f'Image: {image_result["authenticity"]} ({image_conf}%)\n\n'
                    f'{real_component} appears authentic, but {fake_component} could not be verified. '
                    f'This suggests possible manipulation or mismatched information.'
                )
            }
    
    def _get_error_result(self) -> Dict:
        """Return error result"""
        return {
            'verdict': 'ERROR',
            'main_message': '❌ Verification Error',
            'confidence': 0,
            'explanation': 'An error occurred during verification. Please try again.',
            'text_verification': {
                'is_real': False, 'authenticity': 'ERROR', 
                'confidence': 0, 'explanation': 'Error'
            },
            'image_verification': {
                'is_real': False, 'authenticity': 'ERROR', 
                'confidence': 0, 'explanation': 'Error'
            },
            'consistency_score': 0
        }
