from collections import OrderedDict
import numpy as np
from scipy.spatial import distance
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class TrackedObject:
    id: int
    positions: List[Tuple[int, int]]
    disappeared: int

class CentroidTracker:
    def __init__(self, max_disappeared: int = 30):
        self.next_object_id = 0
        self.objects: Dict[int, TrackedObject] = OrderedDict()
        self.max_disappeared = max_disappeared
        
    def register(self, centroid: Tuple[int, int]) -> None:
        """Register a new object with a unique ID."""
        self.objects[self.next_object_id] = TrackedObject(
            id=self.next_object_id,
            positions=[centroid],
            disappeared=0
        )
        self.next_object_id += 1
    
    def deregister(self, object_id: int) -> None:
        """Deregister an object that's been lost for too long."""
        del self.objects[object_id]
    
    def update(self, centroids: List[Tuple[int, int]]) -> Dict[int, TrackedObject]:
        """Update tracked objects with new centroid positions."""
        # Handle the case of no centroids
        if len(centroids) == 0:
            for obj in self.objects.values():
                obj.disappeared += 1
                if obj.disappeared > self.max_disappeared:
                    self.deregister(obj.id)
            return self.objects

        # Initialize arrays for current centroids and tracked objects
        input_centroids = np.array(centroids)
        
        # If we're not tracking any objects, register all centroids
        if len(self.objects) == 0:
            for centroid in centroids:
                self.register(centroid)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = [obj.positions[-1] for obj in self.objects.values()]
            
            # Calculate distances between each pair of tracked/input centroids
            D = distance.cdist(np.array(object_centroids), input_centroids)
            
            # Find the smallest value in each row and sort the row indexes
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            # Loop over the combinations
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                
                object_id = object_ids[row]
                self.objects[object_id].positions.append(centroids[col])
                self.objects[object_id].disappeared = 0
                
                used_rows.add(row)
                used_cols.add(col)
            
            # Handle unused rows and columns
            unused_rows = set(range(D.shape[0])) - used_rows
            unused_cols = set(range(D.shape[1])) - used_cols
            
            # Mark objects as disappeared if their row wasn't used
            for row in unused_rows:
                object_id = object_ids[row]
                self.objects[object_id].disappeared += 1
                if self.objects[object_id].disappeared > self.max_disappeared:
                    self.deregister(object_id)
            
            # Register new objects for unused column centroids
            for col in unused_cols:
                self.register(centroids[col])
        
        return self.objects