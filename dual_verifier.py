"""
Dual Verification Module - AI-POWERED VERSION
Properly uses AI to read, understand, and verify content
"""

from PIL import Image
import numpy as np
from typing import List, Dict

class DualVerifier:
    def __init__(self):
        """Initialize verifier with AI analyzer"""
        print("✓ Initializing AI-Powered Dual Verifier...")
        
        # Import AI analyzer
        try:
            from ai_content_analyzer import AIContentAnalyzer
            self.ai_analyzer = AIContentAnalyzer()
            self.ai_available = True
            print("✓ AI Content Analyzer loaded successfully!")
        except Exception as e:
            print(f"⚠️ AI not available: {e}")
            self.ai_analyzer = None
            self.ai_available = False
        
        # Thresholds
        self.min_evidence_count = 5
        print("✓ Dual Verifier ready!")
    
    def verify_text_and_image(
        self, 
        text: str, 
        user_image: Image.Image,
        text_based_images: List[Dict],
        image_based_images: List[Dict]
    ) -> Dict:
        """
        MAIN VERIFICATION - Uses AI to properly analyze everything
        """
        try:
            if not self.ai_available:
                print("⚠️ AI not available, using basic verification")
                return self._fallback_verification(
                    text, user_image, text_based_images, image_based_images
                )
            
            print("\n" + "="*60)
            print("🤖 STARTING AI-POWERED VERIFICATION")
            print("="*60)
            
            # STEP 1: AI reads and understands TEXT
            print("\n📝 Step 1: AI analyzing text description...")
            text_analysis = self.ai_analyzer.analyze_text_incident(text)
            print(f"   ✓ Identified: {text_analysis.get('incident_type', 'unknown')} in {text_analysis.get('location', 'unknown')}")
            
            # STEP 2: AI reads and understands IMAGE
            print("\n🖼️ Step 2: AI analyzing image content...")
            image_analysis = self.ai_analyzer.analyze_image_content(user_image)
            print(f"   ✓ Scene detected: {image_analysis.get('scene_type', 'unknown')}")
            
            # STEP 3: AI compares if text and image describe SAME incident
            print("\n🔍 Step 3: AI comparing text vs image...")
            comparison = self.ai_analyzer.compare_text_and_image(
                text, user_image, text_analysis, image_analysis
            )
            print(f"   ✓ Match verdict: {comparison.get('verdict', 'UNCERTAIN')}")
            print(f"   ✓ Match score: {comparison.get('match_score', 0)}%")
            
            # STEP 4: AI verifies using web evidence
            print("\n🌐 Step 4: AI verifying with web evidence...")
            print(f"   - Text-based images: {len(text_based_images)}")
            print(f"   - Image-based matches: {len(image_based_images)}")
            
            ai_verification = self.ai_analyzer.verify_with_web_evidence(
                text, user_image, text_based_images, image_based_images
            )
            print(f"   ✓ AI Verdict: {ai_verification.get('verdict', 'UNCERTAIN')}")
            print(f"   ✓ Confidence: {ai_verification.get('confidence', 0)}%")
            
            print("\n" + "="*60)
            print("✅ AI VERIFICATION COMPLETE")
            print("="*60 + "\n")
            
            # STEP 5: Build comprehensive result
            result = self._build_comprehensive_result(
                text, text_analysis, image_analysis, 
                comparison, ai_verification,
                text_based_images, image_based_images
            )
            
            return result
        
        except Exception as e:
            print(f"\n❌ AI verification error: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_verification(
                text, user_image, text_based_images, image_based_images
            )
    
    def verify_text_only(self, text: str, retrieved_images: List[Dict]) -> Dict:
        """Verify text-only with AI"""
        try:
            if not self.ai_available:
                return self._basic_text_verification(text, retrieved_images)
            
            print("🤖 AI analyzing text incident...")
            text_analysis = self.ai_analyzer.analyze_text_incident(text)
            
            ai_result = self.ai_analyzer.verify_with_web_evidence(
                text, None, retrieved_images, []
            )
            
            return {
                'is_real': ai_result['verdict'] in ['REAL', 'LIKELY_REAL'],
                'authenticity': ai_result['verdict'],
                'confidence': ai_result['confidence'],
                'explanation': self._format_text_explanation(text_analysis, ai_result),
                'evidence_count': len(retrieved_images)
            }
        
        except Exception as e:
            print(f"Text verification error: {e}")
            return self._basic_text_verification(text, retrieved_images)
    
    def verify_image_only(self, user_image: Image.Image, retrieved_images: List[Dict]) -> Dict:
        """Verify image-only with AI"""
        try:
            if not self.ai_available:
                return self._basic_image_verification(user_image, retrieved_images)
            
            print("🤖 AI analyzing image content...")
            image_analysis = self.ai_analyzer.analyze_image_content(user_image)
            
            # Use image description for verification
            image_desc = image_analysis.get('description', 'incident scene')
            
            ai_result = self.ai_analyzer.verify_with_web_evidence(
                image_desc, user_image, [], retrieved_images
            )
            
            return {
                'is_real': ai_result['verdict'] in ['REAL', 'LIKELY_REAL'],
                'authenticity': ai_result['verdict'],
                'confidence': ai_result['confidence'],
                'explanation': self._format_image_explanation(image_analysis, ai_result),
                'similar_count': len(retrieved_images)
            }
        
        except Exception as e:
            print(f"Image verification error: {e}")
            return self._basic_image_verification(user_image, retrieved_images)
    
    def _build_comprehensive_result(
        self, text, text_analysis, image_analysis, 
        comparison, ai_verification, text_imgs, img_imgs
    ) -> Dict:
        """
        Build comprehensive result with incident summary
        """
        
        # Extract key information
        incident_type = text_analysis.get('incident_type', 'unknown incident')
        location = text_analysis.get('location', 'unknown location')
        date_time = text_analysis.get('date_time', 'unknown time')
        severity = text_analysis.get('severity', 'unknown')
        
        scene_type = image_analysis.get('scene_type', 'unknown scene')
        image_location = image_analysis.get('location_type', 'unknown')
        
        same_incident = comparison.get('same_incident', False)
        match_score = comparison.get('match_score', 0)
        comparison_reason = comparison.get('reasoning', 'Unable to compare')
        
        ai_verdict = ai_verification.get('verdict', 'UNCERTAIN')
        confidence = ai_verification.get('confidence', 50)
        ai_reasoning = ai_verification.get('reasoning', 'Limited evidence')
        recommendation = ai_verification.get('recommendation', 'Verify from other sources')
        
        # Determine final verdict
        if same_incident and match_score >= 70 and ai_verdict in ['REAL', 'LIKELY_REAL']:
            verdict_type = 'MATCH_AND_REAL'
            main_message = '✅ TEXT and IMAGE MATCH - Both Verified as REAL'
            color_class = 'real'
        
        elif same_incident and ai_verdict == 'UNCERTAIN':
            verdict_type = 'BOTH_REAL_DIFFERENT_INCIDENTS'
            main_message = '⚠️ CAUTION - Possible Mismatch Detected'
            color_class = 'mismatch'
        
        elif not same_incident and match_score < 50:
            verdict_type = 'BOTH_REAL_DIFFERENT_INCIDENTS'
            main_message = '⚠️ MISMATCH - Text and Image Describe Different Things'
            color_class = 'mismatch'
        
        elif ai_verdict in ['LIKELY_FAKE', 'FAKE']:
            verdict_type = 'BOTH_FAKE'
            main_message = '❌ LIKELY FABRICATED - Cannot Verify from Sources'
            color_class = 'fake'
        
        else:
            verdict_type = 'PARTIAL_FAKE'
            main_message = '⚠️ SUSPICIOUS - Partial Verification Only'
            color_class = 'uncertain'
        
        # Build COMPREHENSIVE EXPLANATION with INCIDENT SUMMARY
        explanation = f"""🤖 **AI ANALYSIS RESULTS:**

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 **INCIDENT SUMMARY:**

**Claimed Incident:** {incident_type.upper()}
**Location:** {location}
**Date/Time:** {date_time}
**Severity:** {severity.upper()}
**Key Details:** {text_analysis.get('key_details', 'No details available')[:200]}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📝 **TEXT ANALYSIS:**
- Incident Type: {incident_type}
- Location Mentioned: {location}
- Entities Involved: {text_analysis.get('entities_involved', 'Not specified')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🖼️ **IMAGE ANALYSIS:**
- Scene Type: {scene_type}
- Location Type: {image_location}
- People Visible: {image_analysis.get('people_present', 'unclear')}
- Scene Severity: {image_analysis.get('severity', 'unknown')}
- Description: {image_analysis.get('description', 'N/A')[:150]}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔍 **TEXT vs IMAGE COMPARISON:**
Match Score: {match_score}%
Verdict: {comparison.get('verdict', 'UNCERTAIN')}

{comparison_reason}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🌐 **WEB EVIDENCE VERIFICATION:**
- Images found for text: {len(text_imgs)}
- Similar images found: {len(img_imgs)}
- Total evidence pieces: {len(text_imgs) + len(img_imgs)}

AI Analysis: {ai_reasoning}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ **FINAL VERDICT:** {ai_verdict}
🎯 **CONFIDENCE:** {confidence}%

💡 **RECOMMENDATION:** {recommendation}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        return {
            'verdict': verdict_type,
            'main_message': main_message,
            'confidence': confidence,
            'explanation': explanation,
            'text_verification': {
                'is_real': ai_verdict in ['REAL', 'LIKELY_REAL'],
                'authenticity': ai_verdict,
                'confidence': confidence,
                'explanation': f"{incident_type} in {location}"
            },
            'image_verification': {
                'is_real': ai_verdict in ['REAL', 'LIKELY_REAL'],
                'authenticity': ai_verdict,
                'confidence': confidence,
                'explanation': f"{scene_type} at {image_location}"
            },
            'consistency_score': match_score / 100.0,
            'incident_summary': {
                'type': incident_type,
                'location': location,
                'date_time': date_time,
                'severity': severity,
                'details': text_analysis.get('key_details', '')
            }
        }
    
    def _format_text_explanation(self, text_analysis, ai_result):
        """Format text-only explanation"""
        return f"""📝 **TEXT ANALYSIS:**
- Incident: {text_analysis.get('incident_type', 'unknown')}
- Location: {text_analysis.get('location', 'unknown')}
- Details: {text_analysis.get('key_details', 'N/A')[:150]}

🌐 **VERIFICATION:** {ai_result.get('reasoning', 'Limited evidence')}

💡 **RECOMMENDATION:** {ai_result.get('recommendation', 'Verify from other sources')}
"""
    
    def _format_image_explanation(self, image_analysis, ai_result):
        """Format image-only explanation"""
        return f"""🖼️ **IMAGE ANALYSIS:**
- Scene: {image_analysis.get('scene_type', 'unknown')}
- Location: {image_analysis.get('location_type', 'unknown')}
- Description: {image_analysis.get('description', 'N/A')[:150]}

🌐 **VERIFICATION:** {ai_result.get('reasoning', 'Limited evidence')}

💡 **RECOMMENDATION:** {ai_result.get('recommendation', 'Verify from other sources')}
"""
    
    def _fallback_verification(self, text, image, text_imgs, img_imgs):
        """Fallback when AI is not available"""
        total = len(text_imgs) + len(img_imgs)
        
        if total >= 10:
            verdict = 'MATCH_AND_REAL'
            message = '✅ Both Likely Real (Basic Mode)'
            confidence = 65
        elif total >= 5:
            verdict = 'PARTIAL_FAKE'
            message = '⚠️ Limited Evidence'
            confidence = 45
        else:
            verdict = 'BOTH_FAKE'
            message = '❌ Insufficient Evidence'
            confidence = 25
        
        explanation = f"""⚠️ **AI NOT AVAILABLE - Using Basic Verification**

Found {total} total images online.
- Text-based: {len(text_imgs)}
- Image-based: {len(img_imgs)}

💡 For accurate results, please configure Gemini API key.
"""
        
        return {
            'verdict': verdict,
            'main_message': message,
            'confidence': confidence,
            'explanation': explanation,
            'text_verification': {
                'is_real': total >= 5,
                'authenticity': 'UNCERTAIN',
                'confidence': confidence,
                'explanation': 'AI unavailable - basic check only'
            },
            'image_verification': {
                'is_real': total >= 5,
                'authenticity': 'UNCERTAIN',
                'confidence': confidence,
                'explanation': 'AI unavailable - basic check only'
            },
            'consistency_score': 0.5
        }
    
    def _basic_text_verification(self, text, images):
        """Basic text verification without AI"""
        count = len(images)
        return {
            'is_real': count >= 5,
            'authenticity': 'LIKELY_REAL' if count >= 5 else 'UNCERTAIN',
            'confidence': min(count * 10, 70),
            'explanation': f'Found {count} images. AI unavailable.',
            'evidence_count': count
        }
    
    def _basic_image_verification(self, image, images):
        """Basic image verification without AI"""
        count = len(images)
        return {
            'is_real': count >= 3,
            'authenticity': 'LIKELY_REAL' if count >= 3 else 'UNCERTAIN',
            'confidence': min(count * 15, 70),
            'explanation': f'Found {count} similar images. AI unavailable.',
            'similar_count': count
        }
