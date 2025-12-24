"""
Explanation Generator Module
Generates comprehensive incident descriptions and reports
"""

from typing import Dict
from datetime import datetime

class ExplanationGenerator:
    def __init__(self):
        """Initialize explanation generator"""
        self.templates = {
            'fire': "A fire incident occurred{location}, involving {description}. The event was characterized by flames, smoke, and potential property damage.",
            'flood': "A flooding event was reported{location}. {description} Water levels rose significantly, causing potential infrastructure damage.",
            'accident': "A vehicular or transportation accident occurred{location}. {description} Emergency response teams were likely deployed.",
            'protest': "A public demonstration or protest took place{location}. {description} Citizens gathered to express their views.",
            'explosion': "An explosion was reported{location}. {description} The blast may have caused structural damage and injuries.",
            'natural_disaster': "A natural disaster struck{location}. {description} The event caused widespread impact on the affected area.",
            'violence': "A violent incident occurred{location}. {description} Law enforcement and emergency services responded.",
            'rescue': "A rescue operation was conducted{location}. {description} Emergency personnel worked to save lives.",
            'other': "An incident was reported{location}. {description} Further details are being investigated."
        }
    
    def generate_incident_report(
        self, 
        text_info: Dict, 
        verification_result: Dict,
        mode: str = 'text_to_image'
    ) -> str:
        """Generate comprehensive incident report"""
        try:
            # Header
            report = "=" * 60 + "\n"
            report += "INCIDENT VERIFICATION REPORT\n"
            report += "=" * 60 + "\n\n"
            
            # Timestamp
            report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            report += f"Verification Mode: {mode.upper().replace('_', ' ')}\n\n"
            
            # Authenticity Status
            report += "--- AUTHENTICITY ASSESSMENT ---\n"
            report += f"Status: {verification_result['authenticity']}\n"
            report += f"Confidence Score: {verification_result['confidence']}%\n"
            report += f"Explanation: {verification_result['explanation']}\n\n"
            
            # Incident Details
            if mode == 'text_to_image':
                report += "--- INCIDENT DESCRIPTION ---\n"
                report += f"Original Text: {text_info.get('original_text', 'N/A')}\n\n"
                
                report += "Event Classification:\n"
                report += f"  • Type: {text_info.get('event_type', 'Unknown').upper()}\n"
                report += f"  • Location: {text_info.get('location', 'Unknown')}\n\n"
                
                report += "Extracted Information:\n"
                report += f"  • Keywords: {', '.join(text_info.get('keywords', [])[:8])}\n"
                
                entities = text_info.get('entities', {})
                if entities.get('locations'):
                    report += f"  • Locations: {', '.join(entities['locations'][:3])}\n"
                if entities.get('organizations'):
                    report += f"  • Organizations: {', '.join(entities['organizations'][:3])}\n"
                if entities.get('dates'):
                    report += f"  • Dates: {', '.join(entities['dates'][:3])}\n"
                
                report += "\n"
            
            elif mode == 'image_to_text':
                report += "--- IMAGE ANALYSIS ---\n"
                report += f"Generated Caption: {verification_result.get('caption', 'N/A')}\n\n"
            
            # Evidence
            report += "--- SUPPORTING EVIDENCE ---\n"
            
            if mode == 'text_to_image':
                top_matches = verification_result.get('top_matches', [])
                if top_matches:
                    report += f"Found {len(top_matches)} matching images:\n"
                    for i, match in enumerate(top_matches, 1):
                        report += f"  {i}. {match['name'][:60]}\n"
                        report += f"     Similarity: {match['score']:.4f} | Source: {match['source']}\n"
                else:
                    report += "No matching images found.\n"
            
            elif mode == 'image_to_text':
                similar_images = verification_result.get('similar_images', [])
                if similar_images:
                    report += f"Found {len(similar_images)} similar images online:\n"
                    for i, img in enumerate(similar_images, 1):
                        report += f"  {i}. {img['name'][:60]}\n"
                        report += f"     Similarity: {img['score']:.4f} | Source: {img['source']}\n"
                else:
                    report += "No similar images found online.\n"
            
            report += "\n"
            
            # Detailed Explanation
            report += "--- DETAILED ANALYSIS ---\n"
            
            if verification_result['authenticity'] == 'REAL':
                report += "The verification process found strong evidence supporting the authenticity "
                report += "of this incident. Multiple visual matches were identified, and the described "
                report += "events align with available online imagery.\n\n"
                
                report += "RECOMMENDATION: The incident appears to be genuine based on available evidence. "
                report += "However, always verify through multiple sources for critical decisions.\n"
            
            elif verification_result['authenticity'] == 'UNCERTAIN':
                report += "The verification process found partial evidence but cannot conclusively "
                report += "determine authenticity. Limited visual matches or conflicting information "
                report += "suggests caution.\n\n"
                
                report += "RECOMMENDATION: Further investigation required. Cross-reference with official "
                report += "news sources, government reports, or eyewitness accounts.\n"
            
            else:  # FAKE
                report += "The verification process found minimal or no supporting evidence. The described "
                report += "incident shows low correlation with available online imagery, suggesting potential "
                report += "fabrication or misrepresentation.\n\n"
                
                report += "RECOMMENDATION: Treat with high skepticism. The incident may be false, manipulated, "
                report += "or taken out of context. Do not share without proper verification.\n"
            
            report += "\n" + "=" * 60 + "\n"
            
            return report
        
        except Exception as e:
            print(f"Report generation error: {e}")
            return f"Error generating report: {str(e)}"
    
    def generate_incident_description(
        self, 
        event_type: str, 
        location: str, 
        keywords: list
    ) -> str:
        """Generate natural language incident description"""
        try:
            location_str = f" in {location}" if location != "Unknown" else ""
            
            # Create description from keywords
            keyword_desc = "involving " + ", ".join(keywords[:5])
            
            # Get template
            template = self.templates.get(event_type, self.templates['other'])
            
            # Fill template
            description = template.format(
                location=location_str,
                description=keyword_desc.capitalize()
            )
            
            return description
        except Exception as e:
            print(f"Description generation error: {e}")
            return "An incident was reported. Details are being verified."
    
    def generate_summary(self, verification_result: Dict, text_info: Dict = None) -> str:
        """Generate brief summary for display"""
        try:
            summary = f"**Authenticity:** {verification_result['authenticity']}\n"
            summary += f"**Confidence:** {verification_result['confidence']}%\n\n"
            summary += f"{verification_result['explanation']}\n"
            
            if text_info:
                location = text_info.get('location', 'Unknown')
                event_type = text_info.get('event_type', 'other')
                
                if location != 'Unknown':
                    summary += f"\n**Location:** {location}\n"
                summary += f"**Event Type:** {event_type.replace('_', ' ').title()}\n"
            
            return summary
        except Exception as e:
            print(f"Summary generation error: {e}")
            return "Unable to generate summary"