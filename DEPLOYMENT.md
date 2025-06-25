Deployment Guide for AreYouSecure (AYS)
Quick Start
Local Development
# Clone the repository
git clone https://github.com/YOUR_USERNAME/areyousecure-ays.git
cd areyousecure-ays
# Install dependencies
pip install -r requirements.txt
# Install Tesseract OCR (system dependency)
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr tesseract-ocr-hin tesseract-ocr-tam
# Run the application
export SESSION_SECRET="your-secret-key-here"
gunicorn --bind 0.0.0.0:5000 --reload main:app
Production Deployment
Heroku
# Install Heroku CLI, then:
heroku create your-app-name
heroku buildpacks:add --index 1 heroku-community/apt
echo "tesseract-ocr tesseract-ocr-eng tesseract-ocr-hin tesseract-ocr-tam" > Aptfile
git add . && git commit -m "Add Aptfile for Tesseract"
heroku config:set SESSION_SECRET="your-secret-key"
git push heroku main
Railway
Connect your GitHub repository
Add environment variable: SESSION_SECRET
Deploy automatically
Replit
Import from GitHub
Run: pip install -r requirements.txt
Start with: gunicorn --bind 0.0.0.0:5000 main:app
System Requirements
Python 3.11+
Tesseract OCR with Indian language packs
512MB+ RAM (4MB file processing)
OpenCV system libraries
Environment Variables
SESSION_SECRET: Flask session secret key (required)
DATABASE_URL: PostgreSQL connection (optional)
File Structure After Deployment
areyousecure-ays/
├── app.py                 # Main application
├── main.py               # Entry point
├── requirements.txt      # Dependencies
├── README.md            # Documentation
├── LICENSE              # MIT License
├── .gitignore          # Git ignore rules
├── DEPLOYMENT.md       # This file
├── utils/
│   ├── face_matcher.py
│   ├── image_processor.py
│   └── ocr_processor.py
├── templates/
│   ├── home.html
│   ├── about.html
│   ├── index.html
│   └── result.html
└── static/
    ├── css/custom.css
    └── js/main.js
Features Included
✅ Multi-language OCR (10+ Indian languages)
✅ Live camera functionality
✅ Face detection and comparison
✅ Age verification from Aadhar cards
✅ Mobile-responsive design
✅ Security and privacy features
✅ Professional documentation
