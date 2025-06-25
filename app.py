import os
import logging
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
import cv2
import numpy as np
from datetime import datetime
import uuid
import traceback

from utils.image_processor import ImageProcessor
from utils.face_matcher import FaceMatcher
from utils.ocr_processor import OCRProcessor

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "your-secret-key-here")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}
MAX_FILE_SIZE = 4 * 1024 * 1024  # Further reduced to 4MB to prevent memory issues

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Error handler for file size limit
@app.errorhandler(413)
def too_large(e):
    flash('File too large. Maximum size is 4MB.', 'error')
    return redirect(url_for('index')), 413

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize processors
image_processor = ImageProcessor()
face_matcher = FaceMatcher()
ocr_processor = OCRProcessor()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_age(dob_str):
    """Calculate age from date of birth string"""
    try:
        # Try different date formats
        date_formats = [
            '%d/%m/%Y', '%d-%m-%Y', '%d.%m.%Y',
            '%d/%m/%y', '%d-%m-%y', '%d.%m.%y',
            '%Y-%m-%d', '%Y/%m/%d', '%Y.%m.%d'
        ]
        
        dob = None
        for fmt in date_formats:
            try:
                dob = datetime.strptime(dob_str, fmt)
                break
            except ValueError:
                continue
        
        if not dob:
            return None, None
        
        today = datetime.now()
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        is_adult = age >= 18
        
        return age, is_adult
    except Exception as e:
        logging.error(f"Error calculating age: {e}")
        return None, None

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/verify')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/verify', methods=['POST'])
def verify_identity():
    try:
        # Simple file validation without accessing request.files immediately
        aadhar_file = None
        selfie_file = None
        
        try:
            # Check if the required form fields exist
            if 'aadhar_file' in request.files:
                aadhar_file = request.files['aadhar_file']
            if 'selfie_file' in request.files:
                selfie_file = request.files['selfie_file']
                
            if not aadhar_file or not selfie_file:
                flash('Both Aadhar card and selfie images are required.', 'error')
                return redirect(url_for('index'))
                
        except Exception as e:
            app.logger.error(f"Error processing file upload: {str(e)}")
            flash('Error processing file upload. Please try a smaller file size.', 'error')
            return redirect(url_for('index'))
        
        if aadhar_file.filename == '' or selfie_file.filename == '':
            flash('Please select both files.', 'error')
            return redirect(url_for('index'))
        
        if not (allowed_file(aadhar_file.filename) and allowed_file(selfie_file.filename)):
            flash('Invalid file format. Please upload image files only.', 'error')
            return redirect(url_for('index'))
        
        # Generate unique filenames
        aadhar_filename = str(uuid.uuid4()) + '_' + secure_filename(aadhar_file.filename)
        selfie_filename = str(uuid.uuid4()) + '_' + secure_filename(selfie_file.filename)
        
        # Save uploaded files
        aadhar_path = os.path.join(app.config['UPLOAD_FOLDER'], aadhar_filename)
        selfie_path = os.path.join(app.config['UPLOAD_FOLDER'], selfie_filename)
        
        aadhar_file.save(aadhar_path)
        selfie_file.save(selfie_path)
        
        results = {
            'success': False,
            'dob': None,
            'age': None,
            'is_adult': None,
            'face_match': False,
            'similarity_score': 0.0,
            'errors': []
        }
        
        # Process Aadhar card
        try:
            # Extract DOB using OCR
            dob_text = ocr_processor.extract_dob_from_aadhar(aadhar_path)
            if dob_text:
                results['dob'] = dob_text
                age, is_adult = calculate_age(dob_text)
                if age is not None:
                    results['age'] = age
                    results['is_adult'] = is_adult
                else:
                    results['errors'].append('Could not calculate age from extracted DOB')
            else:
                results['errors'].append('Could not extract DOB from Aadhar card')
        except Exception as e:
            logging.error(f"OCR processing error: {e}")
            results['errors'].append(f'OCR processing failed: {str(e)}')
        
        # Process face matching with quality assessment
        try:
            # Assess selfie quality first
            selfie_quality = face_matcher.get_face_quality_score(selfie_path)
            results['selfie_quality'] = selfie_quality
            
            # Extract face from Aadhar
            aadhar_face = face_matcher.extract_face_from_image(aadhar_path)
            if aadhar_face is None:
                results['errors'].append('Could not detect face in Aadhar card')
            
            # Extract face from selfie
            selfie_face = face_matcher.extract_face_from_image(selfie_path)
            if selfie_face is None:
                results['errors'].append('Could not detect face in selfie')
                # Add quality feedback if available
                if selfie_quality and selfie_quality.get('feedback'):
                    results['errors'].extend(selfie_quality['feedback'])
            
            # Compare faces if both were detected
            if aadhar_face is not None and selfie_face is not None:
                similarity_score = face_matcher.compare_faces(aadhar_face, selfie_face)
                results['similarity_score'] = similarity_score
                results['face_match'] = similarity_score > 0.6  # Threshold for match
            
        except Exception as e:
            logging.error(f"Face matching error: {e}")
            results['errors'].append(f'Face matching failed: {str(e)}')
        
        # Determine overall success
        results['success'] = (
            results['dob'] is not None and 
            results['age'] is not None and 
            results['similarity_score'] > 0
        )
        
        # Clean up uploaded files
        try:
            os.remove(aadhar_path)
            os.remove(selfie_path)
        except Exception as e:
            logging.warning(f"Could not clean up files: {e}")
        
        return render_template('result.html', results=results)
        
    except Exception as e:
        logging.error(f"Verification error: {e}")
        logging.error(traceback.format_exc())
        flash(f'An error occurred during verification: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.errorhandler(500)
def internal_error(e):
    logging.error(f"Internal server error: {e}")
    flash('An internal server error occurred. Please try again.', 'error')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
