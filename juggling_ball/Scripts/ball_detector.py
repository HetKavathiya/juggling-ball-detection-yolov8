"""
Ball Detection Module - Detects juggling balls in images/videos
Works even when balls are in user's hands
"""
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict

class IdenticalBallDetector:
    """Detector for identical juggling balls"""
    
    def __init__(self, model_path, confidence_threshold=0.45):
        """
        Initialize detector
        
        Args:
            model_path: Path to trained YOLOv8 model (best.pt)
            confidence_threshold: Minimum confidence for detections
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        print(f"✅ Model loaded: {model_path}")
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect all balls in frame
        
        Args:
            frame: Input image/frame (numpy array)
        
        Returns:
            List of detections with properties:
            [
                {
                    'bbox': [x1, y1, x2, y2],
                    'center': (cx, cy),
                    'confidence': score,
                    'radius': approximate radius
                }
            ]
        """
        # Run inference
        results = self.model(frame, verbose=False)
        detections = []
        
        for result in results:
            for box in result.boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = box.conf[0].cpu().item()
                
                # Filter by confidence threshold
                if confidence < self.confidence_threshold:
                    continue
                
                # Calculate properties
                width = x2 - x1
                height = y2 - y1
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                radius = max(width, height) // 2
                
                detection = {
                    'bbox': [x1, y1, x2, y2],
                    'center': (cx, cy),
                    'confidence': confidence,
                    'radius': radius,
                    'width': width,
                    'height': height
                }
                detections.append(detection)
        
        return detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw detected balls on frame
        
        Args:
            frame: Input image
            detections: List of detected balls
        
        Returns:
            Image with drawn detections
        """
        result = frame.copy()
        
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            cx, cy = det['center']
            radius = det['radius']
            confidence = det['confidence']
            
            # Draw circle around ball
            cv2.circle(result, (cx, cy), radius + 3, (0, 255, 0), 2)
            
            # Draw center point
            cv2.circle(result, (cx, cy), 5, (0, 0, 255), -1)
            
            # Draw bounding box (faint)
            cv2.rectangle(result, (x1, y1), (x2, y2), (100, 100, 100), 1)
            
            # Draw confidence score
            cv2.putText(result, f'{confidence:.2f}', (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        return result
    
    def detect_image(self, image_path: str, output_path: str = None) -> np.ndarray:
        """Detect balls in single image"""
        print(f"📷 Processing: {image_path}")
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"❌ Cannot read image: {image_path}")
            return None
        
        detections = self.detect(image)
        result = self.draw_detections(image, detections)
        
        # Add detection count
        cv2.putText(result, f'Detections: {len(detections)}',
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        if output_path:
            cv2.imwrite(output_path, result)
            print(f"✅ Saved: {output_path}")
        
        return result