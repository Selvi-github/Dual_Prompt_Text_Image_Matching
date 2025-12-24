# 🔍 Dual-Prompt Text–Image Incident Verification System

A production-ready, AI-powered system that verifies incident authenticity using live internet data, CLIP, and BLIP-2 models.

## 🎯 Features

### Mode 1: Text → Image Verification
- Enter any incident description
- System extracts keywords, location, and event type
- Retrieves related images from the internet (live)
- Computes text-image similarity using CLIP
- Generates verification report with confidence score

### Mode 2: Image → Text Verification
- Upload any incident image
- Generates caption using BLIP-2
- Searches for similar images online
- Compares uploaded image with retrieved images
- Determines authenticity with detailed explanation

## 🚀 Key Highlights

✅ **FULLY WORKING** - All code is complete and functional  
✅ **NO PRE-STORED DATASETS** - Everything retrieved live from internet  
✅ **MODULAR ARCHITECTURE** - Clean, maintainable code structure  
✅ **ERROR HANDLING** - Graceful fallbacks and comprehensive try-catch blocks  
✅ **CPU-COMPATIBLE** - Runs on CPU (no GPU required)  
✅ **FREE TOOLS** - Uses only open-source models and APIs

## 📁 Project Structure

```
incident-verification/
├── app.py                      # Streamlit UI
├── text_processor.py           # Text analysis & keyword extraction
├── image_retriever.py          # Live image retrieval from internet
├── verifier.py                 # CLIP + BLIP-2 verification engine
├── explanation_generator.py    # Report generation
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🛠️ Installation

### Step 1: Clone or Create Project Directory

```bash
mkdir incident-verification
cd incident-verification
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** First-time installation will download:
- CLIP model (~350MB)
- BLIP-2 model (~1GB)
- spaCy language model (~50MB)

This may take 10-20 minutes depending on your internet speed.

### Step 4: Download spaCy Model (if not auto-installed)

```bash
python -m spacy download en_core_web_sm
```

## 🎮 Usage

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Using the System

#### **MODE 1: Text → Image Verification**

1. Select "📝 Text → Image" mode
2. Enter incident description (e.g., "Fire at Mumbai chemical factory on December 15, 2024")
3. Click "🔍 Verify Incident"
4. System will:
   - Extract keywords and location
   - Retrieve related images from internet
   - Verify authenticity using CLIP
   - Generate comprehensive report

#### **MODE 2: Image → Text Verification**

1. Select "🖼️ Image → Text" mode
2. Upload incident image (JPG, PNG, WEBP)
3. Click "🔍 Verify Image"
4. System will:
   - Generate image caption using BLIP-2
   - Search for similar images online
   - Compare and verify authenticity
   - Provide detailed analysis

### Optional: Bing API Key

For better image retrieval results, you can provide a Bing Image Search API key:

1. Get free API key from [Microsoft Azure](https://azure.microsoft.com/en-us/services/cognitive-services/bing-image-search-api/)
2. Enter in sidebar "Bing API Key" field
3. Without API key, system uses demo mode with limited functionality

## 🧪 Testing the System

### Test Case 1: Text Verification

**Input:**
```
A major wildfire broke out in Los Angeles, California on December 20, 2024. 
Multiple fire departments responded to contain the blaze affecting residential areas.
```

**Expected Output:**
- Event Type: Fire
- Location: Los Angeles, California
- Authenticity: REAL/UNCERTAIN (depends on retrieved images)
- Top matching images with similarity scores

### Test Case 2: Image Verification

**Input:** Upload any incident image (fire, flood, accident, etc.)

**Expected Output:**
- Generated caption describing the image
- Similar images found online
- Authenticity determination
- Confidence score

## 🔧 Configuration

Edit these parameters in the Streamlit sidebar:

- **Max Images to Retrieve**: 3-15 images (default: 8)
- **Bing API Key**: Optional for production use

## 📊 How It Works

### Architecture Overview

```
User Input (Text/Image)
    ↓
Text Processing / Image Captioning
    ↓
Live Internet Image Retrieval
    ↓
CLIP Similarity Computation
    ↓
Verification & Scoring
    ↓
Report Generation
```

### Core Technologies

1. **CLIP (OpenAI)**: Text-image similarity matching
2. **BLIP-2 (Salesforce)**: Image captioning
3. **spaCy**: Named entity recognition
4. **Sentence Transformers**: Semantic analysis
5. **Streamlit**: Interactive UI

### Verification Logic

- **Similarity > 0.25**: REAL (High confidence)
- **0.15 < Similarity < 0.25**: UNCERTAIN (Moderate confidence)
- **Similarity < 0.15**: FAKE (Low confidence)

Confidence Score = min(100, |similarity| × 200)

## 🚨 Error Handling

The system includes comprehensive error handling:

- **No images found**: Fallback to demo mode
- **Model loading failure**: Clear error messages
- **Network errors**: Graceful retries
- **Invalid input**: User-friendly warnings

## ⚠️ Limitations

1. **Internet Required**: Must have active internet for image retrieval
2. **API Rate Limits**: Free tier APIs have usage limits
3. **Accuracy**: Verification is probabilistic, not definitive
4. **Language**: Primarily optimized for English text
5. **Processing Time**: First run slower due to model downloads

## 🔐 Privacy & Security

- **No Data Storage**: Images retrieved temporarily, not saved
- **No Logging**: User inputs not logged or stored
- **API Keys**: Stored in session only, not persisted

## 🤝 Contributing

To improve the system:

1. Add more image retrieval sources (Pexels, Unsplash with API keys)
2. Implement multi-language support
3. Add more sophisticated NLP for context extraction
4. Integrate news API for cross-verification
5. Add temporal analysis (incident timeline verification)

## 📝 License

This project is open-source and available under the MIT License.

## 🆘 Troubleshooting

### Issue: Models not loading
**Solution:** Ensure stable internet. Delete `~/.cache/huggingface` and retry.

### Issue: "No module named X"
**Solution:** Run `pip install -r requirements.txt` again.

### Issue: Out of memory
**Solution:** Reduce `max_images` parameter or close other applications.

### Issue: Slow processing
**Solution:** First run is slow (model download). Subsequent runs are faster.

## 📧 Support

For issues or questions:
1. Check error messages in terminal
2. Verify all dependencies installed
3. Ensure Python 3.8+ is being used
4. Check internet connectivity

## 🎓 Citation

If you use this system in research, please cite:

```bibtex
@software{incident_verification_2024,
  title={Dual-Prompt Text-Image Incident Verification System},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/incident-verification}
}
```

---

**Built with ❤️ using Python, Streamlit, CLIP, and BLIP-2**