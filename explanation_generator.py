"""
Explanation Generator Module
Generates human-readable explanations for verification results
"""

from typing import Dict, List

class ExplanationGenerator:
    def __init__(self):
        """Initialize explanation generator"""
        print("✓ Explanation Generator initialized")
        
        # Templates for different scenarios
        self.templates = {
            'high_confidence_real': [
                'Strong evidence suggests this incident is authentic.',
                'Multiple credible sources confirm this event.',
                'Verification indicates this is a real incident.'
            ],
            'medium_confidence_real': [
                'Available evidence suggests this incident likely occurred.',
                'Moderate confidence that this event is authentic.',
                'Some supporting evidence found for this incident.'
            ],
            'low_confidence': [
                'Limited evidence available for verification.',
                'Unable to confidently verify this incident.',
                'Insufficient information to confirm authenticity.'
            ],
            'fake': [
                'No credible evidence found to support this claim.',
                'This incident could not be verified from reliable sources.',
                'Available evidence suggests this may be fabricated.'
            ],
            'mismatch': [
                'Text and image appear to describe different events.',
                'Inconsistency detected between text and visual evidence.',
                'The image may not correspond to the described incident.'
            ]
        }
    
    def generate_explanation(self, verification_result: Dict) -> str:
        """
        Generate detailed explanation based on verification result
        """
        try:
            confidence = verification_result.get('confidence', 0)
            verdict = verification_result.get('verdict', 'UNCERTAIN')
            
            # Base explanation
            explanation_parts = []
            
            # Add confidence assessment
            if confidence >= 80:
                explanation_parts.append("🟢 High Confidence Verification")
            elif confidence >= 60:
                explanation_parts.append("🟡 Medium Confidence Verification")
            else:
                explanation_parts.append("🔴 Low Confidence Verification")
            
            # Add verdict explanation
            if verdict == 'MATCH_AND_REAL':
                explanation_parts.append(
                    "Both the text description and image have been verified as authentic. "
                    "They appear to describe the same incident, and supporting evidence was found online."
                )
            
            elif verdict == 'BOTH_REAL_DIFFERENT_INCIDENTS':
                explanation_parts.append(
                    "⚠️ Warning: Both text and image appear authentic, but they may describe different incidents. "
                    "The image could be from a similar but separate event."
                )
            
            elif verdict == 'BOTH_FAKE':
                explanation_parts.append(
                    "Neither the text nor image could be verified from credible sources. "
                    "This incident may be fabricated or poorly documented."
                )
            
            elif verdict == 'PARTIAL_FAKE':
                explanation_parts.append(
                    "Only one component (text or image) could be verified. "
                    "This suggests possible manipulation or mismatched information."
                )
            
            else:
                explanation_parts.append(
                    "Verification results are uncertain. More information may be needed."
                )
            
            # Add evidence summary
            if 'text_verification' in verification_result:
                text_auth = verification_result['text_verification'].get('authenticity', 'UNKNOWN')
                explanation_parts.append(f"\n📝 Text Status: {text_auth}")
            
            if 'image_verification' in verification_result:
                img_auth = verification_result['image_verification'].get('authenticity', 'UNKNOWN')
                explanation_parts.append(f"🖼️ Image Status: {img_auth}")
            
            return "\n".join(explanation_parts)
        
        except Exception as e:
            print(f"Explanation generation error: {e}")
            return "Unable to generate detailed explanation."
    
    def get_recommendations(self, verification_result: Dict) -> List[str]:
        """Get recommendations based on verification result"""
        recommendations = []
        
        confidence = verification_result.get('confidence', 0)
        verdict = verification_result.get('verdict', 'UNCERTAIN')
        
        if confidence < 50:
            recommendations.append("🔍 Consider seeking additional sources for verification")
            recommendations.append("📰 Check reputable news outlets for coverage of this incident")
        
        if verdict == 'BOTH_REAL_DIFFERENT_INCIDENTS':
            recommendations.append("⚠️ Verify that the image actually corresponds to the described event")
            recommendations.append("🕐 Check the dates and locations of both the text and image")
        
        if verdict in ['BOTH_FAKE', 'PARTIAL_FAKE']:
            recommendations.append("❌ Exercise caution before sharing this information")
            recommendations.append("✋ Wait for confirmation from credible sources")
        
        return recommendations
    
    def format_detailed_report(self, verification_result: Dict) -> str:
        """Generate a detailed verification report"""
        try:
            report_lines = []
            
            report_lines.append("=" * 60)
            report_lines.append("INCIDENT VERIFICATION REPORT")
            report_lines.append("=" * 60)
            
            # Main verdict
            verdict = verification_result.get('main_message', 'Unknown')
            confidence = verification_result.get('confidence', 0)
            
            report_lines.append(f"\n📊 VERDICT: {verdict}")
            report_lines.append(f"🎯 CONFIDENCE: {confidence}%")
            
            # Detailed analysis
            report_lines.append("\n" + "-" * 60)
            report_lines.append("DETAILED ANALYSIS")
            report_lines.append("-" * 60)
            
            if 'text_verification' in verification_result:
                tv = verification_result['text_verification']
                report_lines.append(f"\n📝 TEXT VERIFICATION:")
                report_lines.append(f"   Status: {tv.get('authenticity', 'UNKNOWN')}")
                report_lines.append(f"   Confidence: {tv.get('confidence', 0)}%")
                report_lines.append(f"   Details: {tv.get('explanation', 'N/A')}")
            
            if 'image_verification' in verification_result:
                iv = verification_result['image_verification']
                report_lines.append(f"\n🖼️ IMAGE VERIFICATION:")
                report_lines.append(f"   Status: {iv.get('authenticity', 'UNKNOWN')}")
                report_lines.append(f"   Confidence: {iv.get('confidence', 0)}%")
                report_lines.append(f"   Details: {iv.get('explanation', 'N/A')}")
            
            # Recommendations
            recommendations = self.get_recommendations(verification_result)
            if recommendations:
                report_lines.append("\n" + "-" * 60)
                report_lines.append("RECOMMENDATIONS")
                report_lines.append("-" * 60)
                for rec in recommendations:
                    report_lines.append(f"  {rec}")
            
            report_lines.append("\n" + "=" * 60)
            
            return "\n".join(report_lines)
        
        except Exception as e:
            print(f"Report generation error: {e}")
            return "Unable to generate detailed report."
