import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import logging

class ImageProcessor:
    """Handles image preprocessing and enhancement for better OCR and face detection"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def preprocess_for_ocr(self, image_path):
        """
        Preprocess image for better OCR results
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image from {image_path}")
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Morphological operations to clean up the image
            kernel = np.ones((1, 1), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
            
            # Resize if image is too small (improve OCR accuracy)
            height, width = cleaned.shape
            if height < 600 or width < 600:
                scale_factor = max(600 / height, 600 / width)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                cleaned = cv2.resize(cleaned, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            return cleaned
            
        except Exception as e:
            self.logger.error(f"Error preprocessing image for OCR: {e}")
            # Return original image if preprocessing fails
            try:
                img = cv2.imread(image_path)
                return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            except:
                return None
    
    def preprocess_for_face_detection(self, image_path):
        """
        Preprocess image for better face detection
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image from {image_path}")
            
            # Convert to RGB (face_recognition uses RGB)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Enhance image quality
            pil_img = Image.fromarray(rgb_img)
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(pil_img)
            enhanced = enhancer.enhance(1.2)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(1.1)
            
            # Convert back to numpy array
            processed_img = np.array(enhanced)
            
            return processed_img
            
        except Exception as e:
            self.logger.error(f"Error preprocessing image for face detection: {e}")
            # Return original image if preprocessing fails
            try:
                img = cv2.imread(image_path)
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except:
                return None
    
    def enhance_image_quality(self, image_path, output_path=None):
        """
        Enhance overall image quality
        
        Args:
            image_path (str): Path to the input image
            output_path (str, optional): Path to save enhanced image
            
        Returns:
            str: Path to enhanced image
        """
        try:
            # Open image with PIL
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Enhance brightness
                brightness_enhancer = ImageEnhance.Brightness(img)
                img = brightness_enhancer.enhance(1.1)
                
                # Enhance contrast
                contrast_enhancer = ImageEnhance.Contrast(img)
                img = contrast_enhancer.enhance(1.2)
                
                # Enhance color saturation
                color_enhancer = ImageEnhance.Color(img)
                img = color_enhancer.enhance(1.1)
                
                # Apply slight sharpening
                img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
                
                # Save enhanced image
                if output_path:
                    img.save(output_path, quality=95)
                    return output_path
                else:
                    # Save with _enhanced suffix
                    base_path = image_path.rsplit('.', 1)[0]
                    ext = image_path.rsplit('.', 1)[1]
                    enhanced_path = f"{base_path}_enhanced.{ext}"
                    img.save(enhanced_path, quality=95)
                    return enhanced_path
                    
        except Exception as e:
            self.logger.error(f"Error enhancing image quality: {e}")
            return image_path  # Return original path if enhancement fails
    
    def resize_image(self, image_path, max_width=1000, max_height=1000):
        """
        Resize image while maintaining aspect ratio
        
        Args:
            image_path (str): Path to the input image
            max_width (int): Maximum width
            max_height (int): Maximum height
            
        Returns:
            str: Path to resized image
        """
        try:
            with Image.open(image_path) as img:
                # Calculate new dimensions
                width, height = img.size
                
                # Calculate scaling factor
                scale_w = max_width / width
                scale_h = max_height / height
                scale = min(scale_w, scale_h, 1)  # Don't upscale
                
                if scale < 1:
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    
                    # Resize image
                    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    
                    # Save resized image
                    base_path = image_path.rsplit('.', 1)[0]
                    ext = image_path.rsplit('.', 1)[1]
                    resized_path = f"{base_path}_resized.{ext}"
                    resized_img.save(resized_path, quality=95)
                    
                    return resized_path
                else:
                    return image_path
                    
        except Exception as e:
            self.logger.error(f"Error resizing image: {e}")
            return image_path
    
    def detect_and_correct_orientation(self, image_path):
        """
        Detect and correct image orientation
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            str: Path to corrected image
        """
        try:
            from PIL.ExifTags import ORIENTATION
            
            with Image.open(image_path) as img:
                # Get EXIF data
                exif = img._getexif()
                
                if exif is not None:
                    # Look for orientation tag
                    for tag, value in exif.items():
                        if tag == ORIENTATION:
                            # Rotate image based on orientation
                            if value == 3:
                                img = img.rotate(180, expand=True)
                            elif value == 6:
                                img = img.rotate(270, expand=True)
                            elif value == 8:
                                img = img.rotate(90, expand=True)
                            
                            # Save corrected image
                            base_path = image_path.rsplit('.', 1)[0]
                            ext = image_path.rsplit('.', 1)[1]
                            corrected_path = f"{base_path}_corrected.{ext}"
                            img.save(corrected_path, quality=95)
                            
                            return corrected_path
                
                return image_path
                
        except Exception as e:
            self.logger.error(f"Error correcting orientation: {e}")
            return image_path
