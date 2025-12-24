"""
Text Processing Module
Extracts keywords, locations, and event types from incident descriptions
"""

import re
from typing import Dict, List
from sentence_transformers import SentenceTransformer
import spacy
import subprocess
import sys

class TextProcessor:
    def __init__(self):
        """Initialize text processor with NLP models"""
        # Auto-download spaCy model if not available
        self.nlp = self._load_spacy_model()
        
        # Load sentence transformer
        print("Loading sentence transformer model...")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Event type keywords for classification
        self.event_types = {
            'fire': ['fire', 'burning', 'blaze', 'flames', 'smoke'],
            'flood': ['flood', 'flooding', 'water', 'submerged', 'deluge'],
            'accident': ['accident', 'crash', 'collision', 'wreck'],
            'protest': ['protest', 'demonstration', 'rally', 'march'],
            'explosion': ['explosion', 'blast', 'detonation', 'explode'],
            'natural_disaster': ['earthquake', 'tsunami', 'hurricane', 'tornado', 'cyclone'],
            'violence': ['shooting', 'attack', 'violence', 'stabbing'],
            'rescue': ['rescue', '救援', '救助', '救难'],
            'other': []
        }
    
    def _load_spacy_model(self):
        """Load spaCy model with auto-download (Streamlit Cloud compatible)"""
        model_name = "en_core_web_sm"
        
        try:
            print(f"Loading spaCy model '{model_name}'...")
            nlp = spacy.load(model_name)
            print(f"✓ spaCy model loaded successfully!")
            return nlp
        
        except OSError:
            print(f"❌ spaCy model '{model_name}' not found.")
            print(f"📥 Attempting to download spaCy model...")
            
            try:
                # Import spacy CLI for download
                import spacy.cli
                
                # Download the model using spacy's built-in download
                spacy.cli.download(model_name)
                
                # Load the model after download
                nlp = spacy.load(model_name)
                print(f"✓ spaCy model downloaded and loaded successfully!")
                return nlp
            
            except Exception as e:
                print(f"❌ Failed to download spaCy model: {e}")
                print(f"⚠️ Falling back to basic NLP without spaCy...")
                # Return None to use fallback methods
                return None
        
        except Exception as e:
            print(f"❌ Unexpected error loading spaCy: {e}")
            return None
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text"""
        try:
            # If spaCy model is not available, use basic regex extraction
            if self.nlp is None:
                return self._basic_entity_extraction(text)
            
            doc = self.nlp(text)
            entities = {
                'locations': [],
                'organizations': [],
                'persons': [],
                'dates': []
            }
            
            for ent in doc.ents:
                if ent.label_ == "GPE" or ent.label_ == "LOC":
                    entities['locations'].append(ent.text)
                elif ent.label_ == "ORG":
                    entities['organizations'].append(ent.text)
                elif ent.label_ == "PERSON":
                    entities['persons'].append(ent.text)
                elif ent.label_ == "DATE" or ent.label_ == "TIME":
                    entities['dates'].append(ent.text)
            
            return entities
        except Exception as e:
            print(f"Entity extraction error: {e}")
            return self._basic_entity_extraction(text)
    
    def _basic_entity_extraction(self, text: str) -> Dict[str, List[str]]:
        """Fallback entity extraction without spaCy"""
        entities = {
            'locations': [],
            'organizations': [],
            'persons': [],
            'dates': []
        }
        
        # Basic capitalized word extraction (likely proper nouns)
        words = text.split()
        for i, word in enumerate(words):
            if word and word[0].isupper() and len(word) > 2:
                entities['locations'].append(word)
        
        return entities
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extract important keywords from text"""
        try:
            # If spaCy not available, use basic extraction
            if self.nlp is None:
                return self._basic_keyword_extraction(text, top_n)
            
            doc = self.nlp(text.lower())
            
            # Remove stopwords and punctuation, keep nouns, verbs, adjectives
            keywords = [
                token.text for token in doc 
                if not token.is_stop 
                and not token.is_punct 
                and len(token.text) > 2
                and token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN']
            ]
            
            # Get unique keywords
            keywords = list(dict.fromkeys(keywords))
            
            return keywords[:top_n]
        except Exception as e:
            print(f"Keyword extraction error: {e}")
            return self._basic_keyword_extraction(text, top_n)
    
    def _basic_keyword_extraction(self, text: str, top_n: int = 10) -> List[str]:
        """Fallback keyword extraction without spaCy"""
        # Basic stopwords
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 
                    'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
        
        words = text.lower().split()
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        keywords = list(dict.fromkeys(keywords))  # Remove duplicates
        
        return keywords[:top_n]
    
    def classify_event_type(self, text: str) -> str:
        """Classify the type of incident"""
        try:
            text_lower = text.lower()
            
            for event_type, keywords in self.event_types.items():
                if any(keyword in text_lower for keyword in keywords):
                    return event_type
            
            return 'other'
        except Exception as e:
            print(f"Event classification error: {e}")
            return 'other'
    
    def process_text(self, text: str) -> Dict:
        """Main processing pipeline"""
        try:
            entities = self.extract_entities(text)
            keywords = self.extract_keywords(text)
            event_type = self.classify_event_type(text)
            
            # Create search query combining keywords and location
            search_terms = keywords.copy()
            if entities['locations']:
                search_terms.extend(entities['locations'][:2])
            
            search_query = ' '.join(search_terms[:8])
            
            return {
                'original_text': text,
                'keywords': keywords,
                'entities': entities,
                'event_type': event_type,
                'search_query': search_query,
                'location': entities['locations'][0] if entities['locations'] else 'Unknown'
            }
        except Exception as e:
            print(f"Text processing error: {e}")
            return {
                'original_text': text,
                'keywords': text.split()[:10],
                'entities': {'locations': [], 'organizations': [], 'persons': [], 'dates': []},
                'event_type': 'other',
                'search_query': text[:100],
                'location': 'Unknown'
            }
