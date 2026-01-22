"""
Streamlit Application
Enhanced Dual-Input Incident Verification System
Version: 2.0
"""

import streamlit as st
from PIL import Image
import sys
import os

# Import custom modules
from text_processor import TextProcessor
from image_retriever import ImageRetriever
from dual_verifier import DualVerifier
from explanation_generator import ExplanationGenerator

# Page configuration
st.set_page_config(
    page_title="Dual Input Incident Verification",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with Orange Theme
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #ff6600;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
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
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .real {
        background-color: #d4edda;
        border-color: #28a745;
    }
    .fake {
        background-color: #f8d7da;
        border-color: #dc3545;
    }
    .mismatch {
        background-color: #fff3cd;
        border-color: #ff8c00;
    }
    .uncertain {
        background-color: #ffe5cc;
        border-color: #ff6600;
    }
    .image-card {
        border: 1px solid #ff6600;
        border-radius: 8px;
        padding: 8px;
        margin: 5px;
        background-color: #fff;
    }
    .similarity-badge {
        background-color: #ff6600;
        color: white;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: bold;
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
            verifier = DualVerifier()
            explanation_gen = ExplanationGenerator()
        return text_processor, verifier, explanation_gen
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

def main():
    # Header
    st.markdown('<p class="main-header">🔍 Dual-Input Incident Verification System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Verify incidents using Text + Image together or separately</p>', unsafe_allow_html=True)
    
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
    
    max_images = st.sidebar.slider(
        "Max Images to Retrieve",
        min_value=5,
        max_value=20,
        value=10,
        help="Number of images to retrieve for verification from web"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**💡 How it works:**\n\n"
        "**🤖 AI-Powered Analysis:**\n"
        "• AI reads and understands your text\n"
        "• AI analyzes what's in your image\n"
        "• AI compares if they match\n"
        "• AI verifies against web sources\n\n"
        "**Text + Image Together:**\n"
        "• Checks if both describe same incident\n"
        "• Verifies both against web sources\n"
        "• Detects mismatches\n"
        "• Shows proof images\n\n"
        "**Text Only:**\n"
        "• Retrieves images from web\n"
        "• AI verifies incident authenticity\n\n"
        "**Image Only:**\n"
        "• AI analyzes image content\n"
        "• Reverse image search\n"
        "• Verifies authenticity"
    )
    
    # Main input area
    st.header("📝 Enter Incident Information")
    
    # Create two columns for text and image input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Text Description")
        text_input = st.text_area(
            "Describe the incident:",
            height=200,
            placeholder="Example: Chennai floods December 2024 caused severe damage to infrastructure...",
            help="Describe the incident in detail"
        )
    
    with col2:
        st.subheader("🖼️ Incident Image")
        uploaded_file = st.file_uploader(
            "Upload incident image (optional):",
            type=['jpg', 'jpeg', 'png', 'webp'],
            help="Upload an image related to the incident"
        )
        
        if uploaded_file:
            user_image = Image.open(uploaded_file)
            st.image(user_image, caption="Uploaded Image", use_column_width=True)
    
    # Verify button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        verify_button = st.button("🔍 Verify Incident", type="primary", use_container_width=True)
    
    if verify_button:
        # Check what inputs are provided
        has_text = bool(text_input and text_input.strip())
        has_image = uploaded_file is not None
        
        if not has_text and not has_image:
            st.error("❌ Please provide at least TEXT or IMAGE or both")
            return
        
        try:
            # CASE 1: Both Text and Image provided
            if has_text and has_image:
                st.info("🔄 Mode: TEXT + IMAGE Verification")
                
                # Process text
                with st.spinner("Processing text..."):
                    text_info = st.session_state.text_processor.process_text(text_input)
                st.success("✓ Text processed")
                
                # Retrieve images based on text
                with st.spinner("Retrieving images based on TEXT..."):
                    retriever = ImageRetriever()
                    text_based_images = retriever.retrieve_images(
                        text_info['search_query'], 
                        max_images
                    )
                
                if text_based_images:
                    st.success(f"✓ Retrieved {len(text_based_images)} images for text")
                else:
                    st.warning("⚠️ Could not retrieve images for text")
                
                # Retrieve images based on image (AI-enhanced reverse search)
                with st.spinner("🤖 AI-powered reverse image search..."):
                    image_based_images = retriever.retrieve_similar_images(
                        user_image,
                        text_context=text_input,
                        max_images=max_images
                    )
                
                if image_based_images:
                    st.success(f"✓ Retrieved {len(image_based_images)} images for reverse search")
                else:
                    st.warning("⚠️ Could not retrieve images for reverse search")
                
                # Perform dual verification
                with st.spinner("Cross-verifying text and image..."):
                    result = st.session_state.verifier.verify_text_and_image(
                        text_input,
                        user_image,
                        text_based_images,
                        image_based_images
                    )
                
                st.success("✓ Verification complete")
                
                # Display results
                display_dual_verification_results(
                    result,
                    text_based_images,
                    image_based_images,
                    user_image
                )
            
            # CASE 2: Only Text provided
            elif has_text and not has_image:
                st.info("🔄 Mode: TEXT Only Verification")
                
                # Process text
                with st.spinner("Processing text..."):
                    text_info = st.session_state.text_processor.process_text(text_input)
                st.success("✓ Text processed")
                
                # Retrieve images
                with st.spinner("Retrieving images from web..."):
                    retriever = ImageRetriever()
                    retrieved_images = retriever.retrieve_images(
                        text_info['search_query'],
                        max_images
                    )
                
                if retrieved_images:
                    st.success(f"✓ Retrieved {len(retrieved_images)} images")
                    
                    # Verify
                    with st.spinner("Verifying incident..."):
                        result = st.session_state.verifier.verify_text_only(
                            text_input,
                            retrieved_images
                        )
                    
                    st.success("✓ Verification complete")
                    display_text_only_results(result, retrieved_images)
                else:
                    st.error("❌ No images found. Cannot verify.")
            
            # CASE 3: Only Image provided
            elif not has_text and has_image:
                st.info("🔄 Mode: IMAGE Only Verification (AI-Enhanced)")
                
                # Create retriever instance
                retriever = ImageRetriever()
                
                # AI-powered image analysis
                with st.spinner("🤖 Analyzing image with AI..."):
                    image_description = retriever.image_to_text(user_image)
                
                st.success(f"✓ Image analyzed: {image_description}")
                
                # Retrieve similar images using AI
                with st.spinner("🔍 Finding similar images using AI..."):
                    retrieved_images = retriever.retrieve_similar_images(
                        user_image,
                        text_context=None,
                        max_images=max_images
                    )
                
                if retrieved_images:
                    st.success(f"✓ Found {len(retrieved_images)} similar images")
                    
                    # Verify
                    with st.spinner("Verifying image..."):
                        result = st.session_state.verifier.verify_image_only(
                            user_image,
                            retrieved_images
                        )
                    
                    st.success("✓ Verification complete")
                    display_image_only_results(result, retrieved_images, user_image)
                else:
                    st.error("❌ No similar images found. Cannot verify.")
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)


def display_dual_verification_results(result, text_images, image_images, user_image):
    """Display results for Text + Image verification with retrieved images"""
    st.markdown("---")
    st.header("📋 Verification Results")
    
    # Main verdict box
    verdict = result['verdict']
    
    if verdict == 'MATCH_AND_REAL':
        css_class = 'real'
    elif verdict == 'BOTH_REAL_DIFFERENT_INCIDENTS':
        css_class = 'mismatch'
    elif verdict == 'PARTIAL_FAKE':
        css_class = 'fake'
    else:
        css_class = 'uncertain'
    
    st.markdown(
        f'<div class="result-box {css_class}">'
        f'<h2>{result["main_message"]}</h2>'
        f'<h3>Confidence: {result["confidence"]}%</h3>'
        f'<p style="white-space: pre-line;">{result["explanation"]}</p>'
        f'</div>',
        unsafe_allow_html=True
    )
    
    # Show detailed analysis
    st.subheader("📊 Detailed Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📝 Text Verification")
        st.write(f"**Status:** {result['text_verification']['authenticity']}")
        st.write(f"**Confidence:** {result['text_verification']['confidence']}%")
    
    with col2:
        st.markdown("### 🖼️ Image Verification")
        st.write(f"**Status:** {result['image_verification']['authenticity']}")
        st.write(f"**Confidence:** {result['image_verification']['confidence']}%")
    
    # IMPORTANT: Show original images from internet for REAL incidents
    st.markdown("---")
    
    text_is_real = result['text_verification']['is_real']
    image_is_real = result['image_verification']['is_real']
    
    if text_is_real or image_is_real:
        st.header("🌐 Original Images from Internet (Proof)")
        st.info("📸 Below are real images retrieved from news sources to verify this incident")
        
        # Show text-based images if text is real
        if text_is_real and text_images:
            st.subheader("📝 Images Retrieved Based on Text Description")
            st.caption("These images were found online matching your text description")
            
            cols = st.columns(4)
            for i, img_data in enumerate(text_images[:8]):
                with cols[i % 4]:
                    st.markdown('<div class="image-card">', unsafe_allow_html=True)
                    st.image(img_data['image'], use_column_width=True)
                    
                    # Show original incident details
                    if 'original_description' in img_data:
                        st.caption(f"📰 **{img_data['original_description'][:60]}...**")
                    else:
                        st.caption(f"**{img_data['source'][:30]}**")
                    
                    st.caption(f"🔗 {img_data['name'][:50]}...")
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # Show image-based images if image is real
        if image_is_real and image_images:
            st.markdown("---")
            st.subheader("🖼️ Similar Images Found Online (AI-Enhanced Reverse Search)")
            st.caption("These highly similar images were found using AI-powered matching")
            
            cols = st.columns(4)
            for i, img_data in enumerate(image_images[:8]):
                with cols[i % 4]:
                    st.markdown('<div class="image-card">', unsafe_allow_html=True)
                    st.image(img_data['image'], use_column_width=True)
                    
                    # Show similarity score if available
                    if 'match_percentage' in img_data:
                        st.markdown(
                            f'<span class="similarity-badge">Match: {img_data["match_percentage"]}%</span>', 
                            unsafe_allow_html=True
                        )
                    
                    # Show original incident details
                    if 'original_description' in img_data:
                        st.caption(f"📰 **{img_data['original_description'][:60]}...**")
                    else:
                        st.caption(f"**{img_data['source'][:30]}**")
                    
                    st.caption(f"🔗 {img_data['name'][:50]}...")
                    st.markdown('</div>', unsafe_allow_html=True)
        
        st.success("✅ Above images from news sources verify the incident authenticity")
    else:
        st.warning("⚠️ No original images retrieved - both text and image appear to be fabricated")


def display_text_only_results(result, retrieved_images):
    """Display results for text-only verification"""
    st.markdown("---")
    st.header("📋 Verification Results (Text Only)")
    
    css_class = 'real' if result['is_real'] else 'fake'
    
    st.markdown(
        f'<div class="result-box {css_class}">'
        f'<h2>Authenticity: {result["authenticity"]}</h2>'
        f'<h3>Confidence: {result["confidence"]}%</h3>'
        f'<p>{result["explanation"]}</p>'
        f'</div>',
        unsafe_allow_html=True
    )
    
    if result['is_real'] and retrieved_images:
        st.subheader("🖼️ Retrieved Images from Web")
        cols = st.columns(4)
        for i, img_data in enumerate(retrieved_images[:8]):
            with cols[i % 4]:
                st.image(img_data['image'], use_column_width=True)
                st.caption(f"**{img_data['source'][:30]}**")
                st.caption(f"{img_data['name'][:40]}...")


def display_image_only_results(result, retrieved_images, user_image):
    """Display results for image-only verification"""
    st.markdown("---")
    st.header("📋 Verification Results (Image Only)")
    
    css_class = 'real' if result['is_real'] else 'fake'
    
    st.markdown(
        f'<div class="result-box {css_class}">'
        f'<h2>Authenticity: {result["authenticity"]}</h2>'
        f'<h3>Confidence: {result["confidence"]}%</h3>'
        f'<p>{result["explanation"]}</p>'
        f'</div>',
        unsafe_allow_html=True
    )
    
    if result['is_real'] and retrieved_images:
        st.subheader("🔍 Similar Images Found Online (AI-Enhanced)")
        
        # Show similarity scores if available
        has_scores = any('match_percentage' in img for img in retrieved_images)
        
        cols = st.columns(4)
        for i, img_data in enumerate(retrieved_images[:8]):
            with cols[i % 4]:
                st.markdown('<div class="image-card">', unsafe_allow_html=True)
                st.image(img_data['image'], use_column_width=True)
                
                # Show similarity score
                if 'match_percentage' in img_data:
                    st.markdown(
                        f'<span class="similarity-badge">Match: {img_data["match_percentage"]}%</span>',
                        unsafe_allow_html=True
                    )
                
                # Show original incident information
                if 'original_description' in img_data:
                    st.caption(f"📰 **{img_data['original_description'][:60]}...**")
                else:
                    st.caption(f"**{img_data['source'][:30]}**")
                
                st.caption(f"🔗 {img_data['name'][:50]}...")
                st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
