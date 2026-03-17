"""
Complete pipeline for detecting, tracking, and visualizing juggling balls
- Detects balls even in user's hands
- Tracks each ball with unique ID
- Shows path/trajectory for each ball
- Works on videos and images
"""
import cv2
import numpy as np
from ball_detector import IdenticalBallDetector
from ball_tracker import IdenticalBallTracker
from typing import Dict, List
import json
from datetime import datetime
import os

class JugglingAnalysisPipeline:
    """Complete pipeline for juggling ball analysis"""
    
    def __init__(self, model_path: str, num_balls: int = None,
                 confidence_threshold: float = 0.45,
                 max_tracking_distance: float = 80):
        """
        Initialize pipeline
        
        Args:
            model_path: Path to trained YOLOv8 model (best.pt)
            num_balls: Expected number of balls (optional)
            confidence_threshold: Detection threshold
            max_tracking_distance: Max distance for tracking
        """
        self.detector = IdenticalBallDetector(
            model_path, 
            confidence_threshold=confidence_threshold
        )
        self.tracker = IdenticalBallTracker(
            max_distance=max_tracking_distance,
            max_age=20,
            min_hits=2
        )
        self.num_balls = num_balls
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'frames_with_detections': 0,
            'total_detections': 0,
            'unique_track_ids': set(),
            'detection_rates': []
        }
    
    def process_video(self, input_path: str, output_path: str,
                     display: bool = True, save_stats: bool = True) -> Dict:
        """
        Process video: detect balls, track them, show paths
        
        Args:
            input_path: Path to input video
            output_path: Path to save output video
            display: Show real-time display
            save_stats: Save statistics to JSON
        
        Returns:
            Statistics dictionary
        """
        # Open video
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            print(f"❌ Cannot open video: {input_path}")
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n{'='*70}")
        print(f"🎬 JUGGLING BALL ANALYSIS - VIDEO PROCESSING")
        print(f"{'='*70}")
        print(f"📹 Input: {input_path}")
        print(f"📊 Resolution: {width}x{height}")
        print(f"⏱️  FPS: {fps:.2f}")
        print(f"📈 Total Frames: {total_frames}")
        if self.num_balls:
            print(f"🎪 Expected Balls: {self.num_balls}")
        print(f"{'='*70}\n")
        
        # Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Setup output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print(f"❌ Cannot open video writer for: {output_path}")
            cap.release()
            return None
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Detect balls
                detections = self.detector.detect(frame)
                num_detected = len(detections)
                
                # Track balls
                confirmed_tracks = self.tracker.update(detections)
                
                # Update statistics
                self._update_stats(num_detected, confirmed_tracks)
                
                # Draw on frame
                annotated_frame = self._draw_frame(
                    frame, detections, confirmed_tracks
                )
                
                # Write to output
                out.write(annotated_frame)
                
                # Display
                if display:
                    cv2.imshow('🎪 Juggling Ball Analysis', annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n⚠️  Stopping video processing...")
                        break
                
                # Progress
                if frame_count % 30 == 0 or frame_count == total_frames:
                    progress = (frame_count / total_frames) * 100
                    print(f"✅ Frame {frame_count}/{total_frames} ({progress:.1f}%) | "
                          f"Detected: {num_detected} | Tracked: {len(confirmed_tracks)}")
        
        except Exception as e:
            print(f"❌ Error during processing: {e}")
        
        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
        
        # Finalize statistics
        self._finalize_stats(total_frames)
        
        # Save statistics
        if save_stats:
            stats_path = output_path.replace('.mp4', '_stats.json')
            self._save_stats(stats_path)
        
        print(f"\n{'='*70}")
        print(f"✅ VIDEO PROCESSING COMPLETED!")
        print(f"{'='*70}")
        print(f"📁 Output video: {output_path}")
        self._print_stats()
        print(f"{'='*70}\n")
        
        return self.stats
    
    def process_image(self, image_path: str, output_path: str = None) -> np.ndarray:
        """
        Process single image: detect and mark balls
        
        Args:
            image_path: Path to input image
            output_path: Path to save output image
        
        Returns:
            Annotated image
        """
        print(f"\n{'='*70}")
        print(f"📷 JUGGLING BALL ANALYSIS - IMAGE PROCESSING")
        print(f"{'='*70}")
        print(f"📁 Input: {image_path}\n")
        
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"❌ Cannot read image: {image_path}")
            return None
        
        detections = self.detector.detect(image)
        result = self.detector.draw_detections(image, detections)
        
        # Add detection count
        cv2.putText(result, f'Detected Balls: {len(detections)}',
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        if output_path:
            cv2.imwrite(output_path, result)
            print(f"✅ Output: {output_path}")
        
        print(f"📊 Total balls detected: {len(detections)}\n")
        print(f"{'='*70}\n")
        
        return result
    
    def _draw_frame(self, frame: np.ndarray, detections: List[Dict],
                   tracks: Dict) -> np.ndarray:
        """Draw detections and trajectories on frame"""
        result = frame.copy()
        
        # Color palette for different track IDs
        colors = [
            (255, 0, 0),      # Blue
            (0, 255, 0),      # Green
            (0, 0, 255),      # Red
            (255, 255, 0),    # Cyan
            (255, 0, 255),    # Magenta
            (0, 255, 255),    # Yellow
            (128, 0, 0),      # Dark Blue
            (0, 128, 0),      # Dark Green
            (0, 0, 128),      # Dark Red
            (255, 128, 0),    # Orange
            (128, 0, 255),    # Purple
            (0, 128, 255),    # Light Orange
        ]
        
        # Draw each track with trajectory
        for track_id, track in tracks.items():
            color = colors[track_id % len(colors)]
            history = track['history']
            
            # Draw trajectory path (with fade effect)
            if len(history) > 1:
                for i in range(1, len(history)):
                    pt1 = tuple(map(int, history[i-1]))
                    pt2 = tuple(map(int, history[i]))
                    
                    # Fade effect - older points are dimmer
                    alpha = i / len(history)
                    thickness = max(1, int(2 * alpha))
                    
                    cv2.line(result, pt1, pt2, color, thickness)
            
            # Draw current position with circle
            if len(history) > 0:
                current = tuple(map(int, history[-1]))
                radius = track['radius']
                
                # Filled circle
                cv2.circle(result, current, radius, color, -1)
                # Outline
                cv2.circle(result, current, radius, (255, 255, 255), 2)
                # ID label
                cv2.putText(result, f'{track_id}', 
                           (current[0] - 5, current[1] + 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                           (255, 255, 255), 2)
        
        # Add info panel
        info_text = [
            f"Frame: {self.tracker.frame_count}",
            f"Detected: {len(detections)}",
            f"Tracked: {len(tracks)}"
        ]
        
        y_offset = 30
        for text in info_text:
            cv2.putText(result, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
        
        return result
    
    def _update_stats(self, num_detected: int, tracks: Dict):
        """Update statistics"""
        self.stats['total_frames'] += 1
        self.stats['total_detections'] += num_detected
        
        if num_detected > 0:
            self.stats['frames_with_detections'] += 1
        
        for track_id in tracks.keys():
            self.stats['unique_track_ids'].add(track_id)
        
        if self.num_balls and self.num_balls > 0:
            detection_rate = num_detected / self.num_balls
        else:
            detection_rate = 0
        
        self.stats['detection_rates'].append(detection_rate)
    
    def _finalize_stats(self, total_frames: int):
        """Finalize statistics"""
        if total_frames > 0:
            self.stats['detection_coverage'] = (
                self.stats['frames_with_detections'] / total_frames
            ) * 100
            
            self.stats['avg_detections_per_frame'] = (
                self.stats['total_detections'] / total_frames
            )
            
            if self.stats['detection_rates']:
                self.stats['avg_detection_rate'] = np.mean(
                    self.stats['detection_rates']
                )
    
    def _print_stats(self):
        """Print statistics"""
        print(f"📊 STATISTICS:")
        print(f"   Total Frames: {self.stats['total_frames']}")
        print(f"   Frames with Detections: {self.stats['frames_with_detections']}")
        print(f"   Detection Coverage: {self.stats.get('detection_coverage', 0):.1f}%")
        print(f"   Total Detections: {self.stats['total_detections']}")
        print(f"   Avg Detections/Frame: {self.stats.get('avg_detections_per_frame', 0):.2f}")
        print(f"   Unique Track IDs: {len(self.stats['unique_track_ids'])}")
        if self.num_balls:
            print(f"   Avg Detection Rate: {self.stats.get('avg_detection_rate', 0)*100:.1f}%")
    
    def _save_stats(self, output_path: str):
        """Save statistics to JSON"""
        stats_to_save = {
            'timestamp': datetime.now().isoformat(),
            'num_balls': self.num_balls,
            'total_frames': self.stats['total_frames'],
            'frames_with_detections': self.stats['frames_with_detections'],
            'detection_coverage': self.stats.get('detection_coverage', 0),
            'total_detections': self.stats['total_detections'],
            'avg_detections_per_frame': self.stats.get('avg_detections_per_frame', 0),
            'unique_tracks': len(self.stats['unique_track_ids']),
            'avg_detection_rate': self.stats.get('avg_detection_rate', 0),
        }
        
        with open(output_path, 'w') as f:
            json.dump(stats_to_save, f, indent=2)
        
        print(f"📊 Stats saved: {output_path}")