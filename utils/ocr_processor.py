import pytesseract
import cv2
import re
from datetime import datetime
import logging
import numpy as np
from PIL import Image

class OCRProcessor:
    """Handles OCR processing for extracting text from Aadhar cards"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Configure Tesseract for better OCR results with multi-language support
        self.tesseract_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789/-.'
        self.tesseract_config_multilang = r'--oem 3 --psm 6 -l eng+hin+tel+tam+mar+ben+guj+kan+mal+pan+ori'
        
        # Date patterns for different formats
        self.date_patterns = [
            r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b',  # DD/MM/YYYY or DD-MM-YYYY
            r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{2})\b',   # DD/MM/YY or DD-MM-YY
            r'\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b',   # YYYY/MM/DD or YYYY-MM-DD
            r'\b(\d{2})[/-](\d{2})[/-](\d{4})\b',       # MM/DD/YYYY or MM-DD-YYYY
        ]
        
        # Keywords that might appear near DOB in Aadhar cards (multiple languages)
        self.dob_keywords = [
            # English
            'DOB', 'Date of Birth', 'Birth', 'Born',
            # Hindi
            'जन्म', 'तारीख', 'जन्म तिथि', 'जन्मदिन',
            # Telugu
            'జన్మ', 'తేదీ', 'జన్మ తేదీ',
            # Tamil
            'பிறந்த', 'தேதி', 'பிறந்த தேதி',
            # Marathi
            'जन्म', 'तारीख', 'जन्मतारीख',
            # Bengali
            'জন্ম', 'তারিখ', 'জন্ম তারিখ',
            # Gujarati
            'જન્મ', 'તારીખ', 'જન્મ તારીખ',
            # Kannada
            'ಜನ್ಮ', 'ದಿನಾಂಕ', 'ಜನ್ಮ ದಿನಾಂಕ',
            # Malayalam
            'ജന്മ', 'തീയതി', 'ജന്മ തീയതി',
            # Punjabi
            'ਜਨਮ', 'ਤਾਰੀਖ', 'ਜਨਮ ਤਾਰੀਖ',
            # Odia
            'ଜନ୍ମ', 'ତାରିଖ', 'ଜନ୍ମ ତାରିଖ'
        ]
    
    def extract_text_from_image(self, image_path):
        """
        Extract all text from an image using OCR
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            str: Extracted text
        """
        try:
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Preprocess image for better OCR
            processed_image = self._preprocess_for_ocr(image)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(processed_image)
            
            # Try multi-language OCR first
            try:
                text = pytesseract.image_to_string(pil_image, config=self.tesseract_config_multilang)
                if text.strip():
                    return text.strip()
            except:
                pass
            
            # Fallback to English-only OCR
            text = pytesseract.image_to_string(pil_image, config=self.tesseract_config)
            
            return text.strip()
            
        except Exception as e:
            self.logger.error(f"Error extracting text from image {image_path}: {e}")
            return ""
    
    def extract_dob_from_aadhar(self, image_path):
        """
        Extract Date of Birth from Aadhar card image
        
        Args:
            image_path (str): Path to the Aadhar card image
            
        Returns:
            str or None: Extracted DOB in DD/MM/YYYY format or None if not found
        """
        try:
            # Extract text from image
            text = self.extract_text_from_image(image_path)
            
            if not text:
                self.logger.warning("No text extracted from image")
                return None
            
            self.logger.debug(f"Extracted text: {text}")
            
            # Find DOB using various methods
            dob = self._find_dob_in_text(text)
            
            if dob:
                # Standardize date format
                standardized_dob = self._standardize_date_format(dob)
                self.logger.info(f"Found DOB: {standardized_dob}")
                return standardized_dob
            
            # Try with different preprocessing if first attempt fails
            dob = self._extract_dob_with_enhanced_preprocessing(image_path)
            
            if dob:
                standardized_dob = self._standardize_date_format(dob)
                self.logger.info(f"Found DOB with enhanced preprocessing: {standardized_dob}")
                return standardized_dob
            
            self.logger.warning("Could not extract DOB from Aadhar card")
            return None
            
        except Exception as e:
            self.logger.error(f"Error extracting DOB from Aadhar card: {e}")
            return None
    
    def _preprocess_for_ocr(self, image):
        """
        Preprocess image for better OCR results
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Apply morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Resize image if too small
        height, width = cleaned.shape
        if height < 600:
            scale_factor = 600 / height
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            cleaned = cv2.resize(cleaned, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        return cleaned
    
    def _find_dob_in_text(self, text):
        """
        Find Date of Birth in extracted text
        
        Args:
            text (str): Extracted text from image
            
        Returns:
            str or None: Found date string or None
        """
        # Split text into lines
        lines = text.split('\n')
        
        # Look for DOB keywords first
        for line in lines:
            line = line.strip()
            if any(keyword.lower() in line.lower() for keyword in self.dob_keywords):
                # Extract date from this line
                date_match = self._extract_date_from_line(line)
                if date_match:
                    return date_match
        
        # If no keyword found, search all lines for date patterns
        for line in lines:
            line = line.strip()
            date_match = self._extract_date_from_line(line)
            if date_match:
                # Validate if this looks like a birth date
                if self._is_valid_birth_date(date_match):
                    return date_match
        
        return None
    
    def _extract_date_from_line(self, line):
        """
        Extract date from a text line using regex patterns
        
        Args:
            line (str): Text line to search
            
        Returns:
            str or None: Found date string or None
        """
        for pattern in self.date_patterns:
            match = re.search(pattern, line)
            if match:
                return match.group(0)
        
        return None
    
    def _is_valid_birth_date(self, date_str):
        """
        Validate if a date string could be a valid birth date
        
        Args:
            date_str (str): Date string to validate
            
        Returns:
            bool: True if valid birth date, False otherwise
        """
        try:
            # Try to parse the date
            parsed_date = self._parse_date_string(date_str)
            if not parsed_date:
                return False
            
            # Check if date is reasonable for a birth date
            current_year = datetime.now().year
            birth_year = parsed_date.year
            
            # Birth year should be between 1900 and current year
            if birth_year < 1900 or birth_year > current_year:
                return False
            
            # Age should be reasonable (0-150 years)
            age = current_year - birth_year
            if age < 0 or age > 150:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _parse_date_string(self, date_str):
        """
        Parse date string into datetime object
        
        Args:
            date_str (str): Date string to parse
            
        Returns:
            datetime or None: Parsed datetime object or None
        """
        # Try different date formats
        date_formats = [
            '%d/%m/%Y', '%d-%m-%Y', '%d.%m.%Y',
            '%d/%m/%y', '%d-%m-%y', '%d.%m.%y',
            '%Y/%m/%d', '%Y-%m-%d', '%Y.%m.%d',
            '%m/%d/%Y', '%m-%d-%Y', '%m.%d.%Y'
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return None
    
    def _standardize_date_format(self, date_str):
        """
        Standardize date format to DD/MM/YYYY
        
        Args:
            date_str (str): Input date string
            
        Returns:
            str: Standardized date string
        """
        try:
            parsed_date = self._parse_date_string(date_str)
            if parsed_date:
                return parsed_date.strftime('%d/%m/%Y')
            return date_str
        except Exception:
            return date_str
    
    def _extract_dob_with_enhanced_preprocessing(self, image_path):
        """
        Try extracting DOB with enhanced image preprocessing
        
        Args:
            image_path (str): Path to the image
            
        Returns:
            str or None: Extracted DOB or None
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            
            # Try different preprocessing techniques
            preprocessing_methods = [
                self._preprocess_method_1,
                self._preprocess_method_2,
                self._preprocess_method_3
            ]
            
            for method in preprocessing_methods:
                try:
                    processed_image = method(image)
                    pil_image = Image.fromarray(processed_image)
                    
                    # Try different Tesseract configurations
                    configs = [
                        r'--oem 3 --psm 6',
                        r'--oem 3 --psm 7',
                        r'--oem 3 --psm 8',
                        r'--oem 1 --psm 6'
                    ]
                    
                    for config in configs:
                        text = pytesseract.image_to_string(pil_image, config=config)
                        dob = self._find_dob_in_text(text)
                        
                        if dob:
                            return dob
                            
                except Exception as e:
                    self.logger.debug(f"Preprocessing method failed: {e}")
                    continue
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in enhanced DOB extraction: {e}")
            return None
    
    def _preprocess_method_1(self, image):
        """Alternative preprocessing method 1"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(filtered)
        
        # Apply threshold
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh
    
    def _preprocess_method_2(self, image):
        """Alternative preprocessing method 2"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply sharpening kernel
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(blurred, -1, kernel)
        
        # Apply threshold
        _, thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh
    
    def _preprocess_method_3(self, image):
        """Alternative preprocessing method 3"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(
            opened, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
            cv2.THRESH_BINARY, 15, 10
        )
        
        return thresh
    
    def extract_other_details(self, image_path):
        """
        Extract other details from Aadhar card (name, Aadhar number, etc.)
        
        Args:
            image_path (str): Path to the Aadhar card image
            
        Returns:
            dict: Dictionary containing extracted details
        """
        try:
            text = self.extract_text_from_image(image_path)
            
            details = {
                'name': self._extract_name(text),
                'aadhar_number': self._extract_aadhar_number(text),
                'address': self._extract_address(text)
            }
            
            return details
            
        except Exception as e:
            self.logger.error(f"Error extracting other details: {e}")
            return {}
    
    def _extract_name(self, text):
        """Extract name from text"""
        # This is a simplified implementation
        # In practice, you'd need more sophisticated logic
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if len(line) > 3 and line.isalpha():
                return line
        return None
    
    def _extract_aadhar_number(self, text):
        """Extract Aadhar number from text"""
        # Aadhar number pattern: 12 digits, often separated by spaces
        pattern = r'\b(\d{4})\s*(\d{4})\s*(\d{4})\b'
        match = re.search(pattern, text)
        if match:
            return f"{match.group(1)} {match.group(2)} {match.group(3)}"
        return None
    
    def _extract_address(self, text):
        """Extract address from text"""
        # This is a simplified implementation
        # Address extraction would require more sophisticated NLP
        lines = text.split('\n')
        address_lines = []
        
        for line in lines:
            line = line.strip()
            if len(line) > 10 and not line.isdigit():
                address_lines.append(line)
        
        return '\n'.join(address_lines) if address_lines else None
