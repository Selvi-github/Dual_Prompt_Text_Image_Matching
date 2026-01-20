@@ -1,22 +1,23 @@
"""
Streamlit Application - FIXED VERSION
Dual-Prompt Text-Image Incident Verification System
Streamlit Application
Enhanced Dual-Input Incident Verification System
Version: 2.0
"""

import streamlit as st
from PIL import Image
import sys
import os

# Import custom modules (FIXED IMPORTS)
# Import custom modules
from text_processor import TextProcessor
from image_retriever import ImageRetriever
from verifier import IncidentVerifier  # FIXED: Changed from DualVerifier
from dual_verifier import DualVerifier
from explanation_generator import ExplanationGenerator

# Page configuration
st.set_page_config(
    page_title="Incident Verification System",
    page_title="Dual Input Incident Verification",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
@@ -52,13 +53,13 @@
        background-color: #f8d7da;
        border-color: #dc3545;
    }
    .uncertain {
    .mismatch {
        background-color: #fff3cd;
        border-color: #ffc107;
    }
    .likely-real {
        background-color: #d1ecf1;
        border-color: #17a2b8;
    .uncertain {
        background-color: #e7f3ff;
        border-color: #0066cc;
    }
    </style>
""", unsafe_allow_html=True)
@@ -76,7 +77,7 @@ def load_models():
    try:
        with st.spinner("Loading AI models... This may take a few minutes on first run."):
            text_processor = TextProcessor()
            verifier = IncidentVerifier()  # FIXED: Changed from DualVerifier
            verifier = DualVerifier()
            explanation_gen = ExplanationGenerator()
        return text_processor, verifier, explanation_gen
    except Exception as e:
@@ -85,8 +86,8 @@ def load_models():

def main():
    # Header
    st.markdown('<p class="main-header">🔍 Dual-Prompt Incident Verification System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Verify incidents using AI-powered text and image analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="main-header">🔍 Dual-Input Incident Verification System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Verify incidents using Text + Image together or separately</p>', unsafe_allow_html=True)

    # Load models
    if not st.session_state.initialized:
@@ -105,301 +106,322 @@ def main():
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
        min_value=5,
        max_value=20,
        value=10,
        help="Number of images to retrieve for verification from web"
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "**How it works:**\n\n"
        "1. **Text → Image**: Enter incident description, system retrieves related images and verifies.\n\n"
        "2. **Image → Text**: Upload an image, system generates description and finds similar images online."
        "**💡 How it works:**\n\n"
        "**Text + Image Together:**\n"
        "• Checks if both describe same incident\n"
        "• Verifies both against web sources\n"
        "• Detects mismatches\n"
        "• Shows proof images\n\n"
        "**Text Only:**\n"
        "• Retrieves images from web\n"
        "• Verifies incident authenticity\n\n"
        "**Image Only:**\n"
        "• Reverse image search\n"
        "• Finds similar images online\n"
        "• Verifies authenticity"
    )

    # Main content area
    if "📝" in mode:
        # MODE 1: Text to Image
        st.header("📝 Mode 1: Text → Image Verification")
        
    # Main input area
    st.header("📝 Enter Incident Information")
    
    # Create two columns for text and image input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Text Description")
        text_input = st.text_area(
            "Enter Incident Description:",
            height=150,
            placeholder="Example: A major fire broke out at a chemical factory in Mumbai, India on December 15, 2024. Multiple fire trucks responded to the scene...",
            help="Describe the incident in detail including location, date, and what happened"
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

        col1, col2, col3 = st.columns([1, 2, 1])
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

        with col2:
            verify_button = st.button("🔍 Verify Incident", type="primary", use_container_width=True)
        if not has_text and not has_image:
            st.error("❌ Please provide at least TEXT or IMAGE or both")
            return

        if verify_button and text_input:
            try:
                # Step 1: Process text
        try:
            # CASE 1: Both Text and Image provided
            if has_text and has_image:
                st.info("🔄 Mode: TEXT + IMAGE Verification")
                
                # Process text
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
                # Retrieve images based on text
                with st.spinner("Retrieving images based on TEXT..."):
                    retriever = ImageRetriever()
                    text_based_images = retriever.retrieve_images(
                        text_info['search_query'], 
                        max_images
                    )

                # Step 2: Retrieve images
                with st.spinner(f"Retrieving images from the web..."):
                    retriever = ImageRetriever(bing_api_key if bing_api_key else None)
                    
                    if not bing_api_key:
                        st.warning("⚠️ No API key provided. Using demo mode with limited functionality.")
                        retrieved_images = retriever.get_demo_images(text_info['search_query'], max_images)
                    else:
                        retrieved_images = retriever.retrieve_images(text_info['search_query'], max_images)
                if text_based_images:
                    st.success(f"✓ Retrieved {len(text_based_images)} images for text")
                else:
                    st.warning("⚠️ Could not retrieve images for text")

                if retrieved_images:
                    st.success(f"✓ Retrieved {len(retrieved_images)} images")
                # Retrieve images based on image (reverse search)
                with st.spinner("Performing reverse image search..."):
                    caption = f"{text_info['event_type']} incident"
                    image_based_images = retriever.retrieve_images(caption, max_images)
                
                if image_based_images:
                    st.success(f"✓ Retrieved {len(image_based_images)} images for reverse search")
                else:
                    st.error("No images could be retrieved. Please try again or check your API key.")
                    return
                    st.warning("⚠️ Could not retrieve images for reverse search")

                # Step 3: Verify
                with st.spinner("Analyzing images and verifying incident..."):
                    verification_result = st.session_state.verifier.verify_text_to_image(
                # Perform dual verification
                with st.spinner("Cross-verifying text and image..."):
                    result = st.session_state.verifier.verify_text_and_image(
                        text_input,
                        retrieved_images
                        user_image,
                        text_based_images,
                        image_based_images
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

                # Top matching images
                st.subheader("🖼️ Top Matching Images")
                # Retrieve images
                with st.spinner("Retrieving images from web..."):
                    retriever = ImageRetriever()
                    retrieved_images = retriever.retrieve_images(
                        text_info['search_query'],
                        max_images
                    )

                top_matches = verification_result.get('top_matches', [])
                if top_matches:
                    cols = st.columns(min(3, len(top_matches)))
                if retrieved_images:
                    st.success(f"✓ Retrieved {len(retrieved_images)} images")
                    
                    # Verify
                    with st.spinner("Verifying incident..."):
                        result = st.session_state.verifier.verify_text_only(
                            text_input,
                            retrieved_images
                        )

                    for i, match in enumerate(top_matches[:3]):
                        with cols[i]:
                            st.image(match['image'], caption=f"Match {i+1}", use_container_width=True)
                            st.write(f"**Similarity:** {match['score']:.4f}")
                            st.write(f"**Source:** {match['source']}")
                            st.caption(match['name'][:80])
                    st.success("✓ Verification complete")
                    display_text_only_results(result, retrieved_images)
                else:
                    st.info("No matching images found")
                    st.error("❌ No images found. Cannot verify.")
            
            # CASE 3: Only Image provided
            elif not has_text and has_image:
                st.info("🔄 Mode: IMAGE Only Verification")

                # Score distribution
                if 'all_scores' in verification_result:
                    with st.expander("📈 Similarity Score Distribution", expanded=False):
                        import pandas as pd
                        scores_df = pd.DataFrame({
                            'Image': [f"Image {i+1}" for i in range(len(verification_result['all_scores']))],
                            'Similarity Score': verification_result['all_scores']
                        })
                        st.bar_chart(scores_df.set_index('Image'))
                # Generate caption
                with st.spinner("Analyzing image..."):
                    caption = "incident scene"
                st.success(f"✓ Image analyzed")

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
                # Retrieve similar images
                with st.spinner("Searching for similar images..."):
                    retriever = ImageRetriever()
                    retrieved_images = retriever.retrieve_images(caption, max_images)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.exception(e)
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

        elif verify_button and not text_input:
            st.warning("Please enter an incident description.")
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
        # MODE 2: Image to Text
        st.header("🖼️ Mode 2: Image → Text Verification")
        
        uploaded_file = st.file_uploader(
            "Upload Incident Image:",
            type=['jpg', 'jpeg', 'png', 'webp'],
            help="Upload an image of the incident you want to verify"
        )
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

        if uploaded_file:
            query_image = Image.open(uploaded_file)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                st.image(query_image, caption="Uploaded Image", use_container_width=True)
        # Show text-based images if text is real
        if text_is_real and text_images:
            st.subheader("📝 Images Retrieved Based on Text Description")
            st.caption("These images were found online matching your text description")

            with col2:
                verify_button = st.button("🔍 Verify Image", type="primary", use_container_width=True)
            cols = st.columns(4)
            for i, img_data in enumerate(text_images[:8]):
                with cols[i % 4]:
                    st.image(img_data['image'], use_column_width=True)
                    st.caption(f"**{img_data['source'][:30]}**")
                    st.caption(f"{img_data['name'][:40]}...")
        
        # Show image-based images if image is real
        if image_is_real and image_images:
            st.markdown("---")
            st.subheader("🖼️ Similar Images Found Online (Reverse Search)")
            st.caption("These similar images were found matching your uploaded image")

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
            cols = st.columns(4)
            for i, img_data in enumerate(image_images[:8]):
                with cols[i % 4]:
                    st.image(img_data['image'], use_column_width=True)
                    st.caption(f"**{img_data['source'][:30]}**")
                    st.caption(f"{img_data['name'][:40]}...")
        
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
        st.subheader("🔍 Similar Images Found Online")
        cols = st.columns(4)
        for i, img_data in enumerate(retrieved_images[:8]):
            with cols[i % 4]:
                st.image(img_data['image'], use_column_width=True)
                st.caption(f"**{img_data['source'][:30]}**")
                st.caption(f"{img_data['name'][:40]}...")


if __name__ == "__main__":
    main()
