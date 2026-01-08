"""
Streamlit Application - FIXED VERSION
Dual-Prompt Text-Image Incident Verification System
"""

import streamlit as st
from PIL import Image
import sys
import os

# Import custom modules (FIXED IMPORTS)
from text_processor import TextProcessor
from image_retriever import ImageRetriever
from verifier import IncidentVerifier  # FIXED: Changed from DualVerifier
from explanation_generator import ExplanationGenerator

# Page configuration
st.set_page_config(
    page_title="Incident Verification System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #ddd;
        margin: 1rem 0;
    }
    .real {
        background-color: #d4edda;
        border-color: #28a745;
    }
    .fake {
        background-color: #f8d7da;
        border-color: #dc3545;
    }
    .uncertain {
        background-color: #fff3cd;
        border-color: #ffc107;
    }
    .likely-real {
        background-color: #d1ecf1;
        border-color: #17a2b8;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.text_processor = None
    st.session_state.verifier = None
    st.session_state.explanation_gen = None

@st.cache_resource
def load_models():
    """Load models with caching"""
    try:
        with st.spinner("Loading AI models... This may take a few minutes on first run."):
            text_processor = TextProcessor()
            verifier = IncidentVerifier()  # FIXED: Changed from DualVerifier
            explanation_gen = ExplanationGenerator()
        return text_processor, verifier, explanation_gen
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

def main():
    # Header
    st.markdown('<p class="main-header">🔍 Dual-Prompt Incident Verification System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Verify incidents using AI-powered text and image analysis</p>', unsafe_allow_html=True)
    
    # Load models
    if not st.session_state.initialized:
        text_processor, verifier, explanation_gen = load_models()
        
        if text_processor and verifier and explanation_gen:
            st.session_state.text_processor = text_processor
            st.session_state.verifier = verifier
            st.session_state.explanation_gen = explanation_gen
            st.session_state.initialized = True
            st.success("✓ Models loaded successfully!")
        else:
            st.error("Failed to load models. Please restart the application.")
            return
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")
    
    mode = st.sidebar.radio(
        "Select Verification Mode:",
        ["📝 Text → Image", "🖼️ Image → Text"],
        help="Choose how you want to verify the incident"
    )
    
    st.sidebar.markdown("---")
    
    # API Key input (optional)
    bing_api_key = st.sidebar.text_input(
        "Bing API Key (Optional)",
        type="password",
        help="Enter your Bing Image Search API key for better results. Leave empty to use demo mode."
    )
    
    max_images = st.sidebar.slider(
        "Max Images to Retrieve",
        min_value=3,
        max_value=15,
        value=8,
        help="Number of images to retrieve for verification"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**How it works:**\n\n"
        "1. **Text → Image**: Enter incident description, system retrieves related images and verifies.\n\n"
        "2. **Image → Text**: Upload an image, system generates description and finds similar images online."
    )
    
    # Main content area
    if "📝" in mode:
        # MODE 1: Text to Image
        st.header("📝 Mode 1: Text → Image Verification")
        
        text_input = st.text_area(
            "Enter Incident Description:",
            height=150,
            placeholder="Example: A major fire broke out at a chemical factory in Mumbai, India on December 15, 2024. Multiple fire trucks responded to the scene...",
            help="Describe the incident in detail including location, date, and what happened"
        )
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            verify_button = st.button("🔍 Verify Incident", type="primary", use_container_width=True)
        
        if verify_button and text_input:
            try:
                # Step 1: Process text
                with st.spinner("Processing text..."):
                    text_info = st.session_state.text_processor.process_text(text_input)
                
                st.success("✓ Text processed")
                
                with st.expander("📊 Extracted Information", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Keywords:**", ", ".join(text_info['keywords'][:8]))
                        st.write("**Event Type:**", text_info['event_type'].replace('_', ' ').title())
                    with col2:
                        st.write("**Location:**", text_info['location'])
                        st.write("**Search Query:**", text_info['search_query'])
                
                # Step 2: Retrieve images
                with st.spinner(f"Retrieving images from the web..."):
                    retriever = ImageRetriever(bing_api_key if bing_api_key else None)
                    
                    if not bing_api_key:
                        st.warning("⚠️ No API key provided. Using demo mode with limited functionality.")
                        retrieved_images = retriever.get_demo_images(text_info['search_query'], max_images)
                    else:
                        retrieved_images = retriever.retrieve_images(text_info['search_query'], max_images)
                
                if retrieved_images:
                    st.success(f"✓ Retrieved {len(retrieved_images)} images")
                else:
                    st.error("No images could be retrieved. Please try again or check your API key.")
                    return
                
                # Step 3: Verify
                with st.spinner("Analyzing images and verifying incident..."):
                    verification_result = st.session_state.verifier.verify_text_to_image(
                        text_input,
                        retrieved_images
                    )
                
                st.success("✓ Verification complete")
                
                # Display results
                st.markdown("---")
                st.header("📋 Verification Results")
                
                # Authenticity box with dynamic styling
                authenticity = verification_result['authenticity']
                
                # Map authenticity to CSS class
                css_class_map = {
                    'REAL': 'real',
                    'LIKELY REAL': 'likely-real',
                    'UNCERTAIN': 'uncertain',
                    'LIKELY FAKE': 'fake'
                }
                css_class = css_class_map.get(authenticity, 'uncertain')
                
                st.markdown(
                    f'<div class="result-box {css_class}">'
                    f'<h2>Authenticity: {authenticity}</h2>'
                    f'<h3>Confidence Score: {verification_result["confidence"]}%</h3>'
                    f'<p>{verification_result["explanation"]}</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
                # Top matching images
                st.subheader("🖼️ Top Matching Images")
                
                top_matches = verification_result.get('top_matches', [])
                if top_matches:
                    cols = st.columns(min(3, len(top_matches)))
                    
                    for i, match in enumerate(top_matches[:3]):
                        with cols[i]:
                            st.image(match['image'], caption=f"Match {i+1}", use_container_width=True)
                            st.write(f"**Similarity:** {match['score']:.4f}")
                            st.write(f"**Source:** {match['source']}")
                            st.caption(match['name'][:80])
                else:
                    st.info("No matching images found")
                
                # Score distribution
                if 'all_scores' in verification_result:
                    with st.expander("📈 Similarity Score Distribution", expanded=False):
                        import pandas as pd
                        scores_df = pd.DataFrame({
                            'Image': [f"Image {i+1}" for i in range(len(verification_result['all_scores']))],
                            'Similarity Score': verification_result['all_scores']
                        })
                        st.bar_chart(scores_df.set_index('Image'))
                
                # Full report
                with st.expander("📄 Full Verification Report", expanded=False):
                    report = st.session_state.explanation_gen.generate_incident_report(
                        text_info,
                        verification_result,
                        mode='text_to_image'
                    )
                    st.text(report)
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Report",
                        data=report,
                        file_name="incident_verification_report.txt",
                        mime="text/plain"
                    )
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.exception(e)
        
        elif verify_button and not text_input:
            st.warning("Please enter an incident description.")
    
    else:
        # MODE 2: Image to Text
        st.header("🖼️ Mode 2: Image → Text Verification")
        
        uploaded_file = st.file_uploader(
            "Upload Incident Image:",
            type=['jpg', 'jpeg', 'png', 'webp'],
            help="Upload an image of the incident you want to verify"
        )
        
        if uploaded_file:
            query_image = Image.open(uploaded_file)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                st.image(query_image, caption="Uploaded Image", use_container_width=True)
            
            with col2:
                verify_button = st.button("🔍 Verify Image", type="primary", use_container_width=True)
            
            if verify_button:
                try:
                    # Step 1: Generate caption
                    with st.spinner("Analyzing image and generating description..."):
                        caption = st.session_state.verifier.generate_caption(query_image)
                    
                    st.success("✓ Image analyzed")
                    st.info(f"**Generated Caption:** {caption}")
                    
                    # Step 2: Search for similar images
                    with st.spinner("Searching for similar images online..."):
                        retriever = ImageRetriever(bing_api_key if bing_api_key else None)
                        
                        if not bing_api_key:
                            st.warning("⚠️ No API key provided. Using demo mode with limited functionality.")
                            retrieved_images = retriever.get_demo_images(caption, max_images)
                        else:
                            retrieved_images = retriever.retrieve_images(caption, max_images)
                    
                    if retrieved_images:
                        st.success(f"✓ Found {len(retrieved_images)} similar images")
                    else:
                        st.error("No similar images found. The image may be unique or fabricated.")
                        return
                    
                    # Step 3: Verify
                    with st.spinner("Comparing with online images..."):
                        verification_result = st.session_state.verifier.verify_image_to_text(
                            query_image,
                            retrieved_images
                        )
                    
                    st.success("✓ Verification complete")
                    
                    # Display results
                    st.markdown("---")
                    st.header("📋 Verification Results")
                    
                    # Authenticity box
                    authenticity = verification_result['authenticity']
                    
                    css_class_map = {
                        'REAL': 'real',
                        'LIKELY REAL': 'likely-real',
                        'UNCERTAIN': 'uncertain',
                        'LIKELY FAKE': 'fake'
                    }
                    css_class = css_class_map.get(authenticity, 'uncertain')
                    
                    st.markdown(
                        f'<div class="result-box {css_class}">'
                        f'<h2>Authenticity: {authenticity}</h2>'
                        f'<h3>Confidence Score: {verification_result["confidence"]}%</h3>'
                        f'<p>{verification_result["explanation"]}</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Similar images
                    st.subheader("🖼️ Similar Images Found Online")
                    
                    similar_images = verification_result.get('similar_images', [])
                    if similar_images:
                        cols = st.columns(min(3, len(similar_images)))
                        
                        for i, img_data in enumerate(similar_images[:3]):
                            with cols[i]:
                                st.image(img_data['image'], caption=f"Similar {i+1}", use_container_width=True)
                                st.write(f"**Similarity:** {img_data['score']:.4f}")
                                st.write(f"**Source:** {img_data['source']}")
                                st.caption(img_data['name'][:80])
                    else:
                        st.info("No similar images found")
                    
                    # Score distribution
                    if 'all_scores' in verification_result:
                        with st.expander("📈 Similarity Score Distribution", expanded=False):
                            import pandas as pd
                            scores_df = pd.DataFrame({
                                'Image': [f"Image {i+1}" for i in range(len(verification_result['all_scores']))],
                                'Similarity Score': verification_result['all_scores']
                            })
                            st.bar_chart(scores_df.set_index('Image'))
                    
                    # Full report
                    with st.expander("📄 Full Verification Report", expanded=False):
                        text_info = {
                            'original_text': caption,
                            'location': 'Derived from image',
                            'event_type': 'image_analysis'
                        }
                        
                        report = st.session_state.explanation_gen.generate_incident_report(
                            text_info,
                            verification_result,
                            mode='image_to_text'
                        )
                        st.text(report)
                        
                        # Download button
                        st.download_button(
                            label="📥 Download Report",
                            data=report,
                            file_name="image_verification_report.txt",
                            mime="text/plain"
                        )
                
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.exception(e)

if __name__ == "__main__":
    main()
