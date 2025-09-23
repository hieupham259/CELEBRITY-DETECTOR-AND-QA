# Image Processing Module

## Overview

The `image_handler.py` module provides face detection functionality for the Celebrity Detector and QA system using OpenCV's computer vision capabilities.

## Function: `process_image(image_file)`

### Purpose

Processes uploaded images to detect and highlight faces, preparing them for celebrity recognition.

### Process Flow

#### 1. **Image Loading**

- Converts uploaded image file to in-memory BytesIO buffer
- Transforms binary data to NumPy array format
- Decodes image using OpenCV for processing

#### 2. **Face Detection**

- Converts color image to grayscale for better detection accuracy
- Utilizes OpenCV's Haar Cascade classifier (`haarcascade_frontalface_default.xml`)
- Scans image at multiple scales to detect faces of various sizes

#### 3. **Face Selection**

- Identifies all detected faces in the image
- Selects the largest face by area (width × height)
- Assumes the largest face is the primary subject

#### 4. **Visual Annotation**

- Draws a green rectangle around the detected face
- Uses coordinates (x, y, width, height) for precise positioning
- Rectangle thickness: 3 pixels for clear visibility

#### 5. **Output Generation**

- Encodes processed image to JPEG format
- Returns processed image bytes and face coordinates

### Return Values

| Condition | Image Bytes | Face Coordinates |
|-----------|-------------|------------------|
| Face detected | Processed image with rectangle | `(x, y, width, height)` |
| No face detected | Original image bytes | `None` |

### Technical Specifications

- **Input Format**: Any image format supported by PIL/OpenCV
- **Processing Format**: BGR color space, grayscale for detection
- **Output Format**: JPEG encoded bytes
- **Detection Parameters**:
  - Scale factor: 1.1 (10% size reduction per scale)
  - Min neighbors: 5 (minimum confirmations for detection)

### Dependencies

```python
import cv2              # Computer vision operations
from io import BytesIO  # In-memory binary operations
import numpy as np      # Numerical array processing
```

### Use Cases

- **Celebrity Recognition**: Prepares face regions for identification
- **Image Preprocessing**: Standardizes input for ML models
- **Quality Assurance**: Validates presence of faces before processing
- **User Interface**: Provides visual feedback with face highlightinge function processes an uploaded image.