# 🔍 Dual-Input Incident Verification System

An AI-powered system to verify incidents using **text descriptions** and **images** - together or separately.

## ✨ Features

### 3 Verification Modes:

1. **Text + Image Together** 
   - Cross-verifies if text and image describe the same incident
   - Detects mismatches between description and visual evidence
   - Retrieves proof from web sources

2. **Text Only**
   - Analyzes text description
   - Retrieves related images from web
   - Verifies incident authenticity

3. **Image Only**
   - Performs reverse image search
   - Finds similar images online
   - Verifies image authenticity

## 🚀 Quick Start

### Online (Streamlit Cloud)
Visit: [Your Streamlit App URL]

### Local Setup

```bash
# Clone repository
git clone https://github.com/Selvi-github/Dual_Prompt_Text_Image_Matching.git
cd Dual_Prompt_Text_Image_Matching

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

## 📋 Requirements

- Python 3.8+
- Internet connection (for web scraping)
- No API keys required!

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **NLP**: Basic text processing (spaCy optional)
- **Web Scraping**: BeautifulSoup4
- **Image Processing**: Pillow, NumPy

## 📁 Project Structure

```
├── app.py                      # Main Streamlit application
├── text_processor.py           # Text analysis module
├── image_retriever.py          # Image retrieval from web
├── dual_verifier.py            # Verification logic
├── explanation_generator.py    # Result explanation
├── requirements.txt            # Python dependencies
└── README.md                   # Documentation
```

## 🎯 How It Works

1. **Input**: Provide text description and/or image
2. **Processing**: AI analyzes and retrieves web evidence
3. **Verification**: Cross-checks against online sources
4. **Results**: Shows authenticity with confidence score and proof

## ⚠️ Limitations

- Requires internet connection
- Accuracy depends on available web sources
- May take longer for complex incidents

## 📝 License

MIT License - Feel free to use and modify!

## 👥 Contributing

Contributions welcome! Please open an issue or submit a PR.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Built with ❤️ using Streamlit**
