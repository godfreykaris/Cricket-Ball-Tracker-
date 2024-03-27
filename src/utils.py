from ultralytics import YOLO
from collections import defaultdict, deque
import torch
import numpy as np
import supervision as sv

from src.view_transformer import ViewTransformer

class CalculateSpeed:
    def __init__(self, video_path, model_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.load_model(model_path)
        self.class_names = self.model.names
        self.speed = 0.0
        
        # for trcking objects using supervision
        self.video_path = video_path
        self.video_info = sv.VideoInfo.from_video_path(video_path=self.video_path)
        self.byte_track = sv.ByteTrack(frame_rate=self.fps)
        self.source = np.array([[0, 0], [0, 0], [0, 0], [-0, 0]]) # polygon zone(bounded area for which our model performs inferences)
        self.polygon_zone = sv.PolygonZone(self.source, frame_resolution_wh=self.video_info.resolution_wh)
        
        self.target_width = 3.03
        self.target_height = 20.12
        
        self.target = np.array(
            [
                [0, 0],
                [self.target_width - 1, 0],
                [self.target_width - 1, self.target_height - 1],
                [0, self.target_height - 1]
            ]
        )

    # load model
    def load_model(self, model_path):
        model = YOLO(model_path)
        model.fuse()
        
        return model

    
    # calculate speed of the ball
    def calculate_speed(self):
                
        frame_generator = sv.get_video_frames_generator(source_path=self.video_path)
        
        view_transformer = ViewTransformer(source=self.source, target=self.target)
        
        coordinates = defaultdict(lambda: deque(maxlen=self.video_info.fps))
        for frame in frame_generator:
            results = self.model(frame)[0]
            
            detections = sv.Detections.from_ultralytics(results)
            #detections = detections[self.polygon_zone.trigger(detections=detections)]
            detections = self.byte_track.update_with_detections(detections=detections)
            
            if detections:            
                points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
                points = view_transformer.transform_points(points=points).astype(int)
                
                labels = []
                for tracker_id, [x, y] in zip(detections.tracker_id, points):
                    coordinates[tracker_id].append((x, y))
                    if len(coordinates[tracker_id]) < self.video_info.fps / 2:
                        labels.append(f"{tracker_id}")
                    else:
                        # Calculate the speed
                        coordinate_start = np.array(coordinates[tracker_id][-1])
                        coordinate_end = np.array(coordinates[tracker_id][0])
                        distance = np.linalg.norm(coordinate_end - coordinate_start)  # Calculate Euclidean distance
                        time = len(coordinates[tracker_id]) / self.video_info.fps
                        speed = distance / time
                        print(f"speed: {speed} meters per second")
