"""
Text Processing Module
Extracts keywords, locations, and event types from incident descriptions
Works with or without spaCy (has fallback methods)
"""

import re
from typing import Dict, List

class TextProcessor:
    def __init__(self):
        """Initialize text processor - minimal dependencies"""
        print("Initializing Text Processor...")
        
        # Try to load spaCy if available
        self.nlp = self._load_spacy_model()
        
        # Event type keywords for classification
        self.event_types = {
            'fire': ['fire', 'burning', 'blaze', 'flames', 'smoke'],
            'flood': ['flood', 'flooding', 'water', 'submerged', 'deluge'],
            'accident': ['accident', 'crash', 'collision', 'wreck'],
            'protest': ['protest', 'demonstration', 'rally', 'march'],
            'explosion': ['explosion', 'blast', 'detonation', 'explode'],
            'natural_disaster': ['earthquake', 'tsunami', 'hurricane', 'tornado', 'cyclone'],
            'violence': ['shooting', 'attack', 'violence', 'stabbing'],
            'rescue': ['rescue', 'emergency', 'relief'],
            'other': []
        }
        
        print("✓ Text Processor initialized!")
    
    def _load_spacy_model(self):
        """Try to load spaCy model, return None if not available"""
        try:
            import spacy
            print("Loading spaCy model 'en_core_web_sm'...")
            nlp = spacy.load("en_core_web_sm")
            print("✓ spaCy model loaded successfully!")
            return nlp
        except Exception as e:
            print(f"⚠️ spaCy not available: {e}")
            print("ℹ️ Using fallback NLP methods...")
            return None
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text"""
        try:
            # If spaCy model is available, use it
            if self.nlp is not None:
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
            else:
                # Fallback method
                return self._basic_entity_extraction(text)
        
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
        
        # Extract capitalized words (likely proper nouns)
        words = text.split()
        for i, word in enumerate(words):
            # Clean word
            clean_word = re.sub(r'[^\w]', '', word)
            
            if clean_word and len(clean_word) > 2 and clean_word[0].isupper():
                # Simple heuristic: if followed by capitalized word, likely location/org
                if i + 1 < len(words):
                    next_word = re.sub(r'[^\w]', '', words[i + 1])
                    if next_word and next_word[0].isupper():
                        entities['locations'].append(f"{clean_word} {next_word}")
                else:
                    entities['locations'].append(clean_word)
        
        # Remove duplicates
        entities['locations'] = list(dict.fromkeys(entities['locations']))[:5]
        
        return entities
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extract important keywords from text"""
        try:
            # If spaCy available, use it
            if self.nlp is not None:
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
            else:
                # Fallback method
                return self._basic_keyword_extraction(text, top_n)
        
        except Exception as e:
            print(f"Keyword extraction error: {e}")
            return self._basic_keyword_extraction(text, top_n)
    
    def _basic_keyword_extraction(self, text: str, top_n: int = 10) -> List[str]:
        """Fallback keyword extraction without spaCy"""
        # Common English stopwords
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 
            'those', 'it', 'its', 'they', 'them', 'their', 'we', 'our', 'you',
            'your', 'he', 'she', 'his', 'her', 'him', 'us', 'am', 'i', 'my', 'me'
        }
        
        # Clean and split text
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Remove stopwords and short words
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        
        # Remove duplicates while preserving order
        keywords = list(dict.fromkeys(keywords))
        
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
            # Fallback to basic processing
            return {
                'original_text': text,
                'keywords': text.split()[:10],
                'entities': {'locations': [], 'organizations': [], 'persons': [], 'dates': []},
                'event_type': 'other',
                'search_query': text[:100],
                'location': 'Unknown'
            }
