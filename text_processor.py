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
        """Load spaCy model with auto-download"""
        model_name = "en_core_web_sm"
        
        try:
            print(f"Loading spaCy model '{model_name}'...")
            nlp = spacy.load(model_name)
            print(f"✓ spaCy model loaded successfully!")
            return nlp
        
        except OSError:
            print(f"❌ spaCy model '{model_name}' not found.")
            print(f"📥 Downloading spaCy model... This may take a minute.")
            
            try:
                # Download the model
                subprocess.check_call(
                    [sys.executable, "-m", "spacy", "download", model_name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                # Load the model after download
                nlp = spacy.load(model_name)
                print(f"✓ spaCy model downloaded and loaded successfully!")
                return nlp
            
            except Exception as e:
                print(f"❌ Failed to download spaCy model: {e}")
                print(f"Please run manually: python -m spacy download {model_name}")
                raise
        
        except Exception as e:
            print(f"❌ Unexpected error loading spaCy: {e}")
            raise
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text"""
        try:
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
            return {'locations': [], 'organizations': [], 'persons': [], 'dates': []}
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extract important keywords from text"""
        try:
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
            return text.split()[:top_n]
    
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
