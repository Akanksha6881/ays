// Main JavaScript file for the verification system

document.addEventListener('DOMContentLoaded', function() {
    // Initialize file upload handlers
    initializeFileUploads();
    
    // Initialize form submission handler
    initializeFormSubmission();
    
    // Initialize drag and drop functionality
    initializeDragAndDrop();
    
    // Initialize camera functionality
    initializeCamera();
});

function initializeFileUploads() {
    const aadharFile = document.getElementById('aadhar_file');
    const selfieFile = document.getElementById('selfie_file');
    
    if (aadharFile) {
        aadharFile.addEventListener('change', function(e) {
            handleFileSelection(e, 'aadhar');
        });
    }
    
    if (selfieFile) {
        selfieFile.addEventListener('change', function(e) {
            handleFileSelection(e, 'selfie');
        });
    }
}

function handleFileSelection(event, type) {
    const file = event.target.files[0];
    const uploadArea = document.getElementById(type + 'Upload');
    const placeholder = uploadArea.querySelector('.upload-placeholder');
    const preview = document.getElementById(type + 'Preview');
    const image = document.getElementById(type + 'Image');
    const fileName = document.getElementById(type + 'FileName');
    const fileSize = document.getElementById(type + 'FileSize');
    
    if (file) {
        // Validate file type
        const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/bmp', 'image/tiff', 'image/webp'];
        if (!allowedTypes.includes(file.type)) {
            showAlert('Invalid file type. Please select an image file.', 'danger');
            event.target.value = '';
            return;
        }
        
        // Validate file size (16MB limit)
        const maxSize = 16 * 1024 * 1024;
        if (file.size > maxSize) {
            showAlert('File size too large. Maximum size is 16MB.', 'danger');
            event.target.value = '';
            return;
        }
        
        // Show preview
        const reader = new FileReader();
        reader.onload = function(e) {
            image.src = e.target.result;
            fileName.textContent = file.name;
            fileSize.textContent = formatFileSize(file.size);
            
            placeholder.style.display = 'none';
            preview.style.display = 'flex';
            preview.classList.add('fade-in');
        };
        reader.readAsDataURL(file);
        
        // Add visual feedback
        uploadArea.classList.add('border-success');
        uploadArea.classList.remove('border-danger');
    } else {
        // Reset to placeholder
        placeholder.style.display = 'block';
        preview.style.display = 'none';
        uploadArea.classList.remove('border-success', 'border-danger');
    }
}

function initializeFormSubmission() {
    const form = document.getElementById('verificationForm');
    const submitBtn = document.getElementById('submitBtn');
    
    if (form && submitBtn) {
        form.addEventListener('submit', function(e) {
            // Validate files are selected
            const aadharFile = document.getElementById('aadhar_file').files[0];
            const selfieFile = document.getElementById('selfie_file').files[0];
            
            if (!aadharFile || !selfieFile) {
                e.preventDefault();
                showAlert('Please select both Aadhar card and selfie images.', 'danger');
                return;
            }
            
            // Show loading state
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
            
            // Add loading class to form
            form.classList.add('loading');
            
            // Show processing message
            showAlert('Processing your images... This may take a few moments.', 'info');
        });
    }
}

function initializeDragAndDrop() {
    const uploadAreas = document.querySelectorAll('.upload-area');
    
    uploadAreas.forEach(uploadArea => {
        const fileInput = uploadArea.querySelector('input[type="file"]');
        
        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, preventDefaults, false);
            document.body.addEventListener(eventName, preventDefaults, false);
        });
        
        // Add visual feedback for drag events
        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => {
                uploadArea.classList.add('drag-over');
            });
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => {
                uploadArea.classList.remove('drag-over');
            });
        });
        
        // Handle dropped files
        uploadArea.addEventListener('drop', function(e) {
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                fileInput.files = files;
                
                // Trigger change event
                const event = new Event('change', { bubbles: true });
                fileInput.dispatchEvent(event);
            }
        });
    });
}

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function showAlert(message, type) {
    // Create alert element
    const alert = document.createElement('div');
    alert.className = `alert alert-${type} alert-dismissible fade show`;
    alert.innerHTML = `
        <i class="fas fa-${getAlertIcon(type)} me-2"></i>
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Insert at the top of the container
    const container = document.querySelector('.container');
    const firstChild = container.firstElementChild;
    container.insertBefore(alert, firstChild);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        if (alert.parentNode) {
            alert.classList.remove('show');
            setTimeout(() => {
                if (alert.parentNode) {
                    alert.remove();
                }
            }, 300);
        }
    }, 5000);
}

function getAlertIcon(type) {
    const icons = {
        'success': 'check-circle',
        'danger': 'exclamation-triangle',
        'warning': 'exclamation-triangle',
        'info': 'info-circle'
    };
    return icons[type] || 'info-circle';
}

// Results page functionality
if (window.location.pathname.includes('/verify')) {
    // Add fade-in animation to results
    const cards = document.querySelectorAll('.card');
    cards.forEach((card, index) => {
        setTimeout(() => {
            card.classList.add('fade-in');
        }, index * 200);
    });
}

// Print functionality
function printResults() {
    // Hide elements that shouldn't be printed
    const noPrintElements = document.querySelectorAll('.no-print, .btn');
    noPrintElements.forEach(el => el.style.display = 'none');
    
    window.print();
    
    // Restore elements after printing
    setTimeout(() => {
        noPrintElements.forEach(el => el.style.display = '');
    }, 1000);
}

// Utility function to validate image files
function validateImageFile(file) {
    const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/bmp', 'image/tiff', 'image/webp'];
    const maxSize = 4 * 1024 * 1024; // 4MB
    
    if (!allowedTypes.includes(file.type)) {
        return { valid: false, error: 'Invalid file type. Please select an image file.' };
    }
    
    if (file.size > maxSize) {
        return { valid: false, error: 'File size too large. Maximum size is 4MB.' };
    }
    
    return { valid: true };
}

// Error handling for network issues
window.addEventListener('error', function(e) {
    console.error('JavaScript error:', e);
    showAlert('An unexpected error occurred. Please try again.', 'danger');
});

// Camera functionality
let cameraStream = null;
let capturedImageBlob = null;

function initializeCamera() {
    const cameraBtn = document.getElementById('camera-btn');
    const uploadBtn = document.getElementById('upload-btn');
    const cameraSection = document.getElementById('camera-section');
    const uploadSection = document.getElementById('upload-section');
    const captureBtn = document.getElementById('capture-btn');
    const retakeBtn = document.getElementById('retake-btn');
    const stopCameraBtn = document.getElementById('stop-camera-btn');
    
    if (!cameraBtn) return; // Not on the upload page
    
    // Toggle between camera and upload modes
    cameraBtn.addEventListener('click', function() {
        cameraBtn.classList.remove('btn-outline-primary');
        cameraBtn.classList.add('btn-primary');
        uploadBtn.classList.remove('btn-secondary', 'active');
        uploadBtn.classList.add('btn-outline-secondary');
        
        cameraSection.style.display = 'block';
        uploadSection.style.display = 'none';
        
        startCamera();
    });
    
    uploadBtn.addEventListener('click', function() {
        uploadBtn.classList.remove('btn-outline-secondary');
        uploadBtn.classList.add('btn-secondary', 'active');
        cameraBtn.classList.remove('btn-primary');
        cameraBtn.classList.add('btn-outline-primary');
        
        cameraSection.style.display = 'none';
        uploadSection.style.display = 'block';
        
        stopCamera();
    });
    
    // Camera controls
    if (captureBtn) captureBtn.addEventListener('click', capturePhoto);
    if (retakeBtn) retakeBtn.addEventListener('click', retakePhoto);
    if (stopCameraBtn) stopCameraBtn.addEventListener('click', stopCamera);
}

async function startCamera() {
    try {
        const video = document.getElementById('camera-video');
        
        // Check if browser supports getUserMedia
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error('Camera not supported by this browser');
        }
        
        // Request camera access
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user' // Front camera for selfies
            }
        });
        
        video.srcObject = cameraStream;
        await video.play();
        
        showAlert('Camera started successfully. Position yourself and click capture.', 'success');
        
    } catch (error) {
        console.error('Error accessing camera:', error);
        let errorMessage = 'Could not access camera. ';
        
        if (error.name === 'NotAllowedError') {
            errorMessage += 'Please allow camera permissions and try again.';
        } else if (error.name === 'NotFoundError') {
            errorMessage += 'No camera found on this device.';
        } else if (error.name === 'NotSupportedError') {
            errorMessage += 'Camera not supported by this browser.';
        } else {
            errorMessage += 'Please check permissions or use file upload.';
        }
        
        showAlert(errorMessage, 'danger');
        
        // Switch back to upload mode
        const uploadBtn = document.getElementById('upload-btn');
        if (uploadBtn) uploadBtn.click();
    }
}

function capturePhoto() {
    const video = document.getElementById('camera-video');
    const canvas = document.getElementById('camera-canvas');
    const capturedImage = document.getElementById('captured-image');
    const cameraPreview = document.getElementById('camera-preview');
    const captureBtn = document.getElementById('capture-btn');
    const retakeBtn = document.getElementById('retake-btn');
    
    if (!video || video.videoWidth === 0 || video.videoHeight === 0) {
        showAlert('Camera not ready. Please wait for video to load.', 'warning');
        return;
    }
    
    // Set canvas dimensions to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Draw video frame to canvas
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    
    // Convert canvas to blob and display
    canvas.toBlob(function(blob) {
        if (!blob) {
            showAlert('Failed to capture photo. Please try again.', 'danger');
            return;
        }
        
        capturedImageBlob = blob;
        
        // Show preview
        const imageUrl = URL.createObjectURL(blob);
        capturedImage.src = imageUrl;
        cameraPreview.style.display = 'block';
        
        // Update UI
        captureBtn.style.display = 'none';
        retakeBtn.style.display = 'inline-block';
        
        // Update form with captured image
        updateFormWithCapturedImage(blob);
        
        showAlert('Photo captured successfully!', 'success');
        
    }, 'image/jpeg', 0.8);
}

function retakePhoto() {
    const cameraPreview = document.getElementById('camera-preview');
    const captureBtn = document.getElementById('capture-btn');
    const retakeBtn = document.getElementById('retake-btn');
    
    // Hide preview and reset UI
    if (cameraPreview) cameraPreview.style.display = 'none';
    if (captureBtn) captureBtn.style.display = 'inline-block';
    if (retakeBtn) retakeBtn.style.display = 'none';
    
    // Clear captured image
    capturedImageBlob = null;
    
    // Clear form
    const selfieInput = document.getElementById('selfie_file');
    if (selfieInput) selfieInput.value = '';
    
    // Hide any existing preview
    const selfiePreview = document.getElementById('selfiePreview');
    if (selfiePreview) {
        selfiePreview.style.display = 'none';
    }
}

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
        
        const video = document.getElementById('camera-video');
        if (video) video.srcObject = null;
        
        // Reset camera UI
        const cameraPreview = document.getElementById('camera-preview');
        const captureBtn = document.getElementById('capture-btn');
        const retakeBtn = document.getElementById('retake-btn');
        
        if (cameraPreview) cameraPreview.style.display = 'none';
        if (captureBtn) captureBtn.style.display = 'inline-block';
        if (retakeBtn) retakeBtn.style.display = 'none';
        
        capturedImageBlob = null;
    }
}

function updateFormWithCapturedImage(blob) {
    try {
        // Create a File object from the blob
        const timestamp = new Date().getTime();
        const file = new File([blob], `selfie-capture-${timestamp}.jpg`, { 
            type: 'image/jpeg',
            lastModified: Date.now()
        });
        
        // Create a new FileList containing our file
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        
        // Update the file input
        const selfieInput = document.getElementById('selfie_file');
        if (selfieInput) {
            selfieInput.files = dataTransfer.files;
            
            // Trigger change event to update UI
            const event = new Event('change', { bubbles: true });
            selfieInput.dispatchEvent(event);
        }
    } catch (error) {
        console.error('Error updating form with captured image:', error);
        showAlert('Error processing captured image. Please try again.', 'danger');
    }
}

// Handle form submission errors
window.addEventListener('unhandledrejection', function(e) {
    console.error('Unhandled promise rejection:', e);
    showAlert('A processing error occurred. Please try again.', 'danger');
});
