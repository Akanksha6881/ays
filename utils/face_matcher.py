import cv2
import numpy as np
from PIL import Image
import logging
import os

class FaceMatcher:
    """Handles face detection, extraction, and comparison using OpenCV"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Load OpenCV's pre-trained face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def extract_face_from_image(self, image_path):
        """
        Extract face features from an image using OpenCV
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            numpy.ndarray or None: Face region array or None if no face found
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"Could not load image: {image_path}")
                return None
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            if len(faces) == 0:
                self.logger.warning(f"No faces found in image: {image_path}")
                return None
            
            if len(faces) > 1:
                self.logger.warning(f"Multiple faces found in image: {image_path}. Using the largest face.")
                # Select the largest face (by area)
                largest_face = max(faces, key=lambda face: face[2] * face[3])
                faces = [largest_face]
            
            # Extract face region
            x, y, w, h = faces[0]
            face_region = gray[y:y+h, x:x+w]
            
            # Resize to standard size for comparison
            face_resized = cv2.resize(face_region, (100, 100))
            
            return face_resized
            
        except Exception as e:
            self.logger.error(f"Error extracting face from image {image_path}: {e}")
            return None
    
    def compare_faces(self, face1, face2, tolerance=0.6):
        """
        Compare two face regions and return similarity score using template matching
        
        Args:
            face1 (numpy.ndarray): First face region
            face2 (numpy.ndarray): Second face region
            tolerance (float): Threshold for face matching
            
        Returns:
            float: Similarity score (0.0 to 1.0)
        """
        try:
            if face1 is None or face2 is None:
                return 0.0
            
            # Ensure both faces are the same size
            if face1.shape != face2.shape:
                face2 = cv2.resize(face2, (face1.shape[1], face1.shape[0]))
            
            # Calculate correlation coefficient
            correlation = cv2.matchTemplate(face1, face2, cv2.TM_CCOEFF_NORMED)[0][0]
            
            # Convert correlation to similarity score (0.0 to 1.0)
            similarity_score = max(0.0, (correlation + 1.0) / 2.0)
            
            return similarity_score
            
        except Exception as e:
            self.logger.error(f"Error comparing faces: {e}")
            return 0.0
    
    def detect_faces_in_image(self, image_path):
        """
        Detect all faces in an image and return their locations
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            list: List of face locations [(x, y, w, h), ...]
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return []
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            return faces.tolist()
            
        except Exception as e:
            self.logger.error(f"Error detecting faces in image {image_path}: {e}")
            return []
    
    def crop_face_from_image(self, image_path, output_path=None, padding=20):
        """
        Extract and crop the face from an image
        
        Args:
            image_path (str): Path to the input image
            output_path (str, optional): Path to save cropped face
            padding (int): Padding around the face
            
        Returns:
            str or None: Path to cropped face image or None if no face found
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            if len(faces) == 0:
                self.logger.warning(f"No faces found in image: {image_path}")
                return None
            
            # Use the largest face if multiple faces found
            if len(faces) > 1:
                largest_face = max(faces, key=lambda face: face[2] * face[3])
                face_location = largest_face
            else:
                face_location = faces[0]
            
            # Extract face coordinates
            x, y, w, h = face_location
            
            # Add padding
            height, width = image.shape[:2]
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(width - x, w + 2 * padding)
            h = min(height - y, h + 2 * padding)
            
            # Crop face
            face_image = image[y:y+h, x:x+w]
            
            # Save cropped face
            if output_path is None:
                base_path = image_path.rsplit('.', 1)[0]
                ext = image_path.rsplit('.', 1)[1]
                output_path = f"{base_path}_face.{ext}"
            
            cv2.imwrite(output_path, face_image)
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error cropping face from image {image_path}: {e}")
            return None
    
    def enhance_face_detection(self, image_path, output_path=None):
        """
        Enhance image specifically for better face detection
        
        Args:
            image_path (str): Path to the input image
            output_path (str, optional): Path to save enhanced image
            
        Returns:
            str: Path to enhanced image
        """
        try:
            # Read image with OpenCV
            img = cv2.imread(image_path)
            
            # Convert to grayscale for histogram equalization
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_gray = clahe.apply(gray)
            
            # Convert back to BGR
            enhanced_img = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
            
            # Apply Gaussian blur to reduce noise
            enhanced_img = cv2.GaussianBlur(enhanced_img, (3, 3), 0)
            
            # Save enhanced image
            if output_path is None:
                base_path = image_path.rsplit('.', 1)[0]
                ext = image_path.rsplit('.', 1)[1]
                output_path = f"{base_path}_face_enhanced.{ext}"
            
            cv2.imwrite(output_path, enhanced_img)
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error enhancing image for face detection: {e}")
            return image_path
    
    def batch_compare_faces(self, reference_face, face_list):
        """
        Compare a reference face with multiple faces
        
        Args:
            reference_face (numpy.ndarray): Reference face region
            face_list (list): List of face regions to compare
            
        Returns:
            list: List of similarity scores
        """
        try:
            if reference_face is None:
                return [0.0] * len(face_list)
            
            similarity_scores = []
            
            for face in face_list:
                if face is not None:
                    score = self.compare_faces(reference_face, face)
                    similarity_scores.append(score)
                else:
                    similarity_scores.append(0.0)
            
            return similarity_scores
            
        except Exception as e:
            self.logger.error(f"Error in batch face comparison: {e}")
            return [0.0] * len(face_list)
    
    def get_face_quality_score(self, image_path):
        """
        Assess the quality of face detection in an image with detailed feedback
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            dict: Quality assessment results with feedback
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {
                    'face_detected': False,
                    'face_count': 0,
                    'face_size': 0,
                    'quality_score': 0.0,
                    'brightness_score': 0.0,
                    'sharpness_score': 0.0,
                    'feedback': ['Could not load image']
                }
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            feedback = []
            
            if len(faces) == 0:
                feedback.append("No face detected. Please ensure face is clearly visible.")
                return {
                    'face_detected': False,
                    'face_count': 0,
                    'face_size': 0,
                    'quality_score': 0.0,
                    'brightness_score': 0.0,
                    'sharpness_score': 0.0,
                    'feedback': feedback
                }
            
            if len(faces) > 1:
                feedback.append("Multiple faces detected. Please ensure only one person is in the image.")
            
            # Calculate face size (as percentage of image)
            face_location = faces[0]
            x, y, w, h = face_location
            
            face_area = w * h
            image_height, image_width = image.shape[:2]
            image_area = image_height * image_width
            
            face_size_ratio = face_area / image_area
            
            # Check face size
            if face_size_ratio < 0.05:
                feedback.append("Face is too small. Please move closer to the camera.")
            elif face_size_ratio > 0.5:
                feedback.append("Face is too large. Please move away from the camera.")
            
            # Assess brightness
            face_roi = gray[y:y+h, x:x+w]
            brightness = np.mean(face_roi)
            brightness_score = 1.0 - abs(brightness - 127) / 127  # Ideal brightness around 127
            
            if brightness < 60:
                feedback.append("Image is too dark. Please ensure good lighting.")
            elif brightness > 200:
                feedback.append("Image is too bright. Please reduce lighting or avoid direct light.")
            
            # Assess sharpness using Laplacian variance
            laplacian = cv2.Laplacian(face_roi, cv2.CV_64F)
            sharpness_score = laplacian.var() / 1000  # Normalize
            sharpness_score = min(1.0, sharpness_score)
            
            if sharpness_score < 0.3:
                feedback.append("Image appears blurry. Please ensure camera is steady and in focus.")
            
            # Calculate overall quality score
            quality_score = (face_size_ratio * 5 + brightness_score + sharpness_score) / 3
            quality_score = min(1.0, quality_score)
            
            if quality_score > 0.7 and not feedback:
                feedback.append("Good quality image!")
            
            return {
                'face_detected': True,
                'face_count': len(faces),
                'face_size': face_size_ratio,
                'quality_score': quality_score,
                'brightness_score': brightness_score,
                'sharpness_score': sharpness_score,
                'feedback': feedback
            }
            
        except Exception as e:
            self.logger.error(f"Error assessing face quality: {e}")
            return {
                'face_detected': False,
                'face_count': 0,
                'face_size': 0,
                'quality_score': 0.0,
                'brightness_score': 0.0,
                'sharpness_score': 0.0,
                'feedback': ['Error analyzing image quality']
            }
