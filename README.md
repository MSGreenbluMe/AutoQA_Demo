# AutoQA - AI Quality Assurance for Call Centers

Automatizované hodnotenie kvality hovorov v call centrách pomocou AI.

## Features

- **Audio Upload** - Podpora WAV, MP3, M4A, FLAC (do 50MB)
- **Transcription** - Automatický prepis cez Gladia.io s identifikáciou hovoriacich
- **AI Analysis** - Analýza kvality cez Google Gemini API
- **Dashboard** - Prehľadné vizualizácie metrík

### Metriky

- Sentiment Analysis
- FCR (First Call Resolution) Prediction
- Speech Metrics (počet slov, rýchlosť reči, pomer času)
- Interruption Detection
- Empathy & Professionalism Scoring
- Overall Quality Score

## Local Development

### Prerequisites

- Python 3.9+
- Gladia.io API key
- Google Gemini API key

### Installation

```bash
# Clone repository
git clone https://github.com/MSGreenbluMe/AutoQA_Demo.git
cd AutoQA_Demo

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your API keys
```

### Run

```bash
streamlit run app.py
```

## Streamlit Cloud Deployment

### 1. Connect Repository

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Select this repository
4. Set main file path: `app.py`

### 2. Configure Secrets

In Streamlit Cloud app settings, add secrets:

```toml
GLADIA_API_KEY = "your_gladia_api_key"
GEMINI_API_KEY = "your_gemini_api_key"
```

### 3. Deploy

Click "Deploy" and wait for the app to build.

## Tech Stack

- **Frontend:** Streamlit
- **Transcription:** Gladia.io API
- **AI Analysis:** Google Gemini API (gemini-2.5-pro)
- **Visualization:** Plotly

## Project Structure

```
AutoQA_Demo/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── .env.example             # API keys template
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── services/
│   ├── gladia.py            # Gladia.io transcription
│   └── gemini.py            # Gemini analysis
├── components/
│   └── dashboard.py         # Dashboard visualizations
└── utils/
    ├── audio_utils.py       # Audio utilities
    └── config.py            # Configuration helper
```

## License

Proprietary - Greenblu.me

---

🐙 **Powered by Greenblu.me** | 2025
