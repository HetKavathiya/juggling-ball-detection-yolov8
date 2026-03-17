"""
Ball Tracking Module - Tracks balls across frames and maintains paths
Uses Hungarian Algorithm for optimal assignment
"""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple

class IdenticalBallTracker:
    """
    Tracker optimized for identical juggling balls
    Assigns unique IDs and tracks paths across frames
    """
    
    def __init__(self, max_distance=80, max_age=20, min_hits=2):
        """
        Initialize tracker
        
        Args:
            max_distance: Max centroid distance to match detections to tracks
            max_age: Max frames to keep a track alive without detections
            min_hits: Min detections before confirming a track
        """
        self.max_distance = max_distance
        self.max_age = max_age
        self.min_hits = min_hits
        
        self.next_track_id = 0
        self.tracks = {}  # {track_id: track_data}
        self.frame_count = 0
    
    def update(self, detections: List[Dict]) -> Dict[int, Dict]:
        """
        Update tracker with detections from new frame
        
        Returns:
            Dictionary of confirmed tracks: {track_id: track_data}
        """
        self.frame_count += 1
        
        # No detections - age all tracks
        if len(detections) == 0:
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['age'] += 1
                if self.tracks[track_id]['age'] > self.max_age:
                    del self.tracks[track_id]
            return self._get_confirmed_tracks()
        
        # Get current detection centers
        current_centers = np.array([det['center'] for det in detections])
        
        # Get existing track centers
        track_ids = list(self.tracks.keys())
        
        if len(track_ids) == 0:
            # Create new tracks for all detections
            for det in detections:
                self._create_track(det)
            return self._get_confirmed_tracks()
        
        # Build cost matrix
        cost_matrix = self._build_cost_matrix(current_centers, track_ids)
        
        # Hungarian algorithm for optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matched_detection_idx = set()
        matched_track_idx = set()
        
        # Process matches
        for row, col in zip(row_ind, col_ind):
            if cost_matrix[row, col] < self.max_distance:
                track_id = track_ids[row]
                detection = detections[col]
                
                # Update track with new detection
                self.tracks[track_id]['history'].append(detection['center'])
                self.tracks[track_id]['bbox'] = detection['bbox']
                self.tracks[track_id]['confidence'] = detection['confidence']
                self.tracks[track_id]['hits'] += 1
                self.tracks[track_id]['age'] = 0  # Reset age
                self.tracks[track_id]['radius'] = detection['radius']
                
                matched_detection_idx.add(col)
                matched_track_idx.add(row)
        
        # Age unmatched tracks
        for track_idx, track_id in enumerate(track_ids):
            if track_idx not in matched_track_idx:
                self.tracks[track_id]['age'] += 1
        
        # Create new tracks for unmatched detections
        for i, det in enumerate(detections):
            if i not in matched_detection_idx:
                self._create_track(det)
        
        # Remove old tracks
        for track_id in list(self.tracks.keys()):
            if self.tracks[track_id]['age'] > self.max_age:
                del self.tracks[track_id]
        
        return self._get_confirmed_tracks()
    
    def _build_cost_matrix(self, current_centers: np.ndarray, 
                          track_ids: List[int]) -> np.ndarray:
        """
        Build cost matrix for Hungarian algorithm
        Considers centroid distance and velocity
        """
        n_tracks = len(track_ids)
        n_detections = len(current_centers)
        
        cost_matrix = np.zeros((n_tracks, n_detections))
        
        for i, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            last_center = track['history'][-1]
            
            # Calculate distances
            distances = np.linalg.norm(
                current_centers - np.array(last_center), axis=1
            )
            
            # Add velocity penalty
            if len(track['history']) > 1:
                velocity = np.array(track['history'][-1]) - np.array(track['history'][-2])
                velocity_penalty = self._velocity_penalty(
                    current_centers, last_center, velocity
                )
                distances = distances + 0.3 * velocity_penalty
            
            cost_matrix[i, :] = distances
        
        return cost_matrix
    
    def _velocity_penalty(self, current_centers: np.ndarray, 
                         last_center: Tuple, velocity: np.ndarray) -> np.ndarray:
        """Calculate penalty for velocity deviation"""
        expected_centers = last_center + velocity
        penalties = np.linalg.norm(current_centers - expected_centers, axis=1)
        return penalties
    
    def _create_track(self, detection: Dict):
        """Create a new track for a detection"""
        self.tracks[self.next_track_id] = {
            'history': [detection['center']],
            'bbox': detection['bbox'],
            'confidence': detection['confidence'],
            'radius': detection['radius'],
            'hits': 1,
            'age': 0
        }
        self.next_track_id += 1
    
    def _get_confirmed_tracks(self) -> Dict[int, Dict]:
        """Return only confirmed tracks (hits >= min_hits)"""
        confirmed = {}
        for track_id, track in self.tracks.items():
            if track['hits'] >= self.min_hits:
                confirmed[track_id] = track
        return confirmed
    
    def get_all_tracks(self) -> Dict[int, Dict]:
        """Return all tracks including unconfirmed ones"""
        return self.tracks.copy()
    
    def reset(self):
        """Reset tracker"""
        self.tracks = {}
        self.next_track_id = 0
        self.frame_count = 0