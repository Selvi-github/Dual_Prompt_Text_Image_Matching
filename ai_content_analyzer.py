"""
AI Content Analyzer - Uses Google Gemini to READ and UNDERSTAND content
Properly analyzes text and images before verification
"""

import google.generativeai as genai
from PIL import Image
import io
import os
from typing import Dict, Optional
import streamlit as st

class AIContentAnalyzer:
    def __init__(self):
        """Initialize AI analyzer with Gemini"""
        self.model = None
        self._initialize_gemini()
    
    def _initialize_gemini(self):
        """Initialize Google Gemini AI (FREE API)"""
        try:
            # Try to get API key from Streamlit secrets first
            if hasattr(st, 'secrets') and 'GEMINI_API_KEY' in st.secrets:
                api_key = st.secrets['GEMINI_API_KEY']
            # Then try environment variable
            elif 'GEMINI_API_KEY' in os.environ:
                api_key = os.environ['GEMINI_API_KEY']
            else:
                print("⚠️ No Gemini API key found. Using fallback analysis.")
                return
            
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            print("✓ AI Content Analyzer initialized with Gemini")
        
        except Exception as e:
            print(f"⚠️ Gemini initialization failed: {e}")
            print("Using fallback analysis methods")
    
    def analyze_text_incident(self, text: str) -> Dict:
        """
        AI analyzes text to extract incident details
        Returns: location, event type, date, description
        """
        try:
            if not self.model:
                return self._fallback_text_analysis(text)
            
            prompt = f"""Analyze this incident description and extract key information:

Text: "{text}"

Extract and return ONLY in this JSON format:
{{
    "incident_type": "type of incident (fire/flood/accident/etc)",
    "location": "where it happened",
    "date_time": "when it happened (if mentioned)",
    "key_details": "brief summary of what happened",
    "severity": "low/medium/high",
    "entities_involved": "people, organizations mentioned"
}}

Be precise and factual. If information is not in the text, write "not specified"."""

            response = self.model.generate_content(prompt)
            
            # Parse AI response
            import json
            try:
                result = json.loads(response.text.strip().replace('```json', '').replace('```', ''))
            except:
                # If JSON parsing fails, extract manually
                result = {
                    "incident_type": "incident",
                    "location": "unknown",
                    "date_time": "not specified",
                    "key_details": text[:200],
                    "severity": "medium",
                    "entities_involved": "not specified"
                }
            
            print(f"✓ AI Text Analysis: {result['incident_type']} in {result['location']}")
            return result
        
        except Exception as e:
            print(f"AI text analysis error: {e}")
            return self._fallback_text_analysis(text)
    
    def analyze_image_content(self, image: Image.Image) -> Dict:
        """
        AI analyzes image to describe what's shown
        Returns: scene description, incident type, objects detected
        """
        try:
            if not self.model:
                return self._fallback_image_analysis(image)
            
            prompt = """Analyze this image and describe what you see in detail.

Focus on:
1. What type of incident or scene is this? (fire, flood, accident, protest, etc)
2. Where does this appear to be? (urban area, forest, building, etc)
3. What key objects or elements are visible?
4. Are there any people, vehicles, or structures?
5. What is the severity/intensity of the scene?

Return ONLY in this JSON format:
{
    "scene_type": "what kind of scene/incident",
    "location_type": "urban/rural/indoor/outdoor/etc",
    "key_objects": "main objects visible",
    "people_present": "yes/no/unclear",
    "severity": "low/medium/high",
    "description": "detailed description of what you see"
}"""

            response = self.model.generate_content([prompt, image])
            
            # Parse response
            import json
            try:
                result = json.loads(response.text.strip().replace('```json', '').replace('```', ''))
            except:
                # Extract from text response
                result = {
                    "scene_type": "incident scene",
                    "location_type": "outdoor",
                    "key_objects": "various elements",
                    "people_present": "unclear",
                    "severity": "medium",
                    "description": response.text[:200]
                }
            
            print(f"✓ AI Image Analysis: {result['scene_type']}")
            return result
        
        except Exception as e:
            print(f"AI image analysis error: {e}")
            return self._fallback_image_analysis(image)
    
    def compare_text_and_image(
        self, 
        text: str, 
        image: Image.Image,
        text_analysis: Optional[Dict] = None,
        image_analysis: Optional[Dict] = None
    ) -> Dict:
        """
        AI compares text and image to check if they describe SAME incident
        Returns: match score, explanation, verdict
        """
        try:
            if not self.model:
                return self._fallback_comparison(text_analysis, image_analysis)
            
            # Analyze if not already done
            if not text_analysis:
                text_analysis = self.analyze_text_incident(text)
            if not image_analysis:
                image_analysis = self.analyze_image_content(image)
            
            prompt = f"""Compare this text description with the image analysis:

TEXT DESCRIPTION: "{text}"

IMAGE ANALYSIS:
- Scene type: {image_analysis.get('scene_type', 'unknown')}
- Location: {image_analysis.get('location_type', 'unknown')}
- Description: {image_analysis.get('description', 'unknown')}

Question: Do the text and image describe the SAME incident?

Consider:
1. Does the incident TYPE match? (fire vs fire, flood vs flood, etc)
2. Does the LOCATION match? (urban vs urban, forest vs forest, etc)
3. Do the DETAILS align?

Return ONLY in this JSON format:
{{
    "same_incident": true/false,
    "match_score": 0-100 (percentage match),
    "reasoning": "detailed explanation why they match or don't match",
    "verdict": "MATCH/MISMATCH/UNCERTAIN"
}}"""

            response = self.model.generate_content(prompt)
            
            # Parse response
            import json
            try:
                result = json.loads(response.text.strip().replace('```json', '').replace('```', ''))
            except:
                # Fallback parsing
                text_lower = response.text.lower()
                same = 'match' in text_lower and 'mismatch' not in text_lower
                
                result = {
                    "same_incident": same,
                    "match_score": 70 if same else 30,
                    "reasoning": response.text[:300],
                    "verdict": "MATCH" if same else "UNCERTAIN"
                }
            
            print(f"✓ AI Comparison: {result['verdict']} ({result['match_score']}%)")
            return result
        
        except Exception as e:
            print(f"AI comparison error: {e}")
            return self._fallback_comparison(text_analysis, image_analysis)
    
    def verify_with_web_evidence(
        self,
        text: str,
        image: Image.Image,
        retrieved_text_images: list,
        retrieved_image_matches: list
    ) -> Dict:
        """
        Final AI-powered verification using all evidence
        """
        try:
            if not self.model:
                return self._basic_verification(
                    len(retrieved_text_images), 
                    len(retrieved_image_matches)
                )
            
            prompt = f"""Verify this incident using available evidence:

CLAIM (Text): "{text}"

EVIDENCE GATHERED:
- {len(retrieved_text_images)} images found for text description
- {len(retrieved_image_matches)} similar images found for the uploaded image

Based on evidence quantity and quality, determine:
1. Is this incident REAL or FAKE?
2. Confidence level (0-100%)
3. Detailed reasoning

Return ONLY JSON:
{{
    "verdict": "REAL/LIKELY_REAL/UNCERTAIN/LIKELY_FAKE/FAKE",
    "confidence": 0-100,
    "reasoning": "detailed explanation",
    "recommendation": "what user should do"
}}

Guidelines:
- REAL: 8+ credible images, high similarity
- LIKELY_REAL: 5-7 images, moderate match
- UNCERTAIN: 3-4 images, low confidence
- LIKELY_FAKE: 1-2 images, poor match
- FAKE: 0 images or no credible sources"""

            response = self.model.generate_content(prompt)
            
            import json
            try:
                result = json.loads(response.text.strip().replace('```json', '').replace('```', ''))
            except:
                result = self._basic_verification(
                    len(retrieved_text_images),
                    len(retrieved_image_matches)
                )
            
            print(f"✓ Final Verification: {result['verdict']} ({result['confidence']}%)")
            return result
        
        except Exception as e:
            print(f"Verification error: {e}")
            return self._basic_verification(
                len(retrieved_text_images),
                len(retrieved_image_matches)
            )
    
    def _fallback_text_analysis(self, text: str) -> Dict:
        """Basic text analysis without AI"""
        words = text.lower().split()
        
        incident_type = "incident"
        if any(w in words for w in ['fire', 'burn', 'blaze']):
            incident_type = "fire"
        elif any(w in words for w in ['flood', 'water', 'rain']):
            incident_type = "flood"
        elif any(w in words for w in ['accident', 'crash', 'collision']):
            incident_type = "accident"
        
        return {
            "incident_type": incident_type,
            "location": "unknown location",
            "date_time": "not specified",
            "key_details": text[:150],
            "severity": "medium",
            "entities_involved": "not specified"
        }
    
    def _fallback_image_analysis(self, image: Image.Image) -> Dict:
        """Basic image analysis without AI"""
        return {
            "scene_type": "incident scene",
            "location_type": "outdoor",
            "key_objects": "various elements",
            "people_present": "unclear",
            "severity": "medium",
            "description": "Scene captured in image"
        }
    
    def _fallback_comparison(self, text_analysis: Dict, image_analysis: Dict) -> Dict:
        """Basic comparison without AI"""
        if not text_analysis or not image_analysis:
            return {
                "same_incident": False,
                "match_score": 40,
                "reasoning": "Insufficient data for comparison",
                "verdict": "UNCERTAIN"
            }
        
        # Simple keyword matching
        text_type = text_analysis.get('incident_type', '').lower()
        image_type = image_analysis.get('scene_type', '').lower()
        
        match = text_type in image_type or image_type in text_type
        
        return {
            "same_incident": match,
            "match_score": 65 if match else 35,
            "reasoning": f"Text describes {text_type}, image shows {image_type}",
            "verdict": "MATCH" if match else "UNCERTAIN"
        }
    
    def _basic_verification(self, text_img_count: int, image_img_count: int) -> Dict:
        """Basic verification without AI"""
        total_evidence = text_img_count + image_img_count
        
        if total_evidence >= 10:
            verdict = "LIKELY_REAL"
            confidence = min(70 + total_evidence, 90)
        elif total_evidence >= 5:
            verdict = "UNCERTAIN"
            confidence = 50
        else:
            verdict = "LIKELY_FAKE"
            confidence = max(20, 40 - total_evidence * 5)
        
        return {
            "verdict": verdict,
            "confidence": confidence,
            "reasoning": f"Found {total_evidence} pieces of evidence online",
            "recommendation": "Verify from additional sources"
        }
