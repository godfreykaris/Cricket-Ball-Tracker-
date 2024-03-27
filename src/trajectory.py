# import supervision as sv
# from ultralytics import YOLO

# model = YOLO("./models/initial_best.pt")
# frame_generator = sv.get_video_frames_generator("./test_videos/1.mp4")
# bounding_box_annotator = sv.BoundingBoxAnnotator()

# for frame in frame_generator:
#     result = model(frame)[0]
#     detections = sv.Detections.from_ultralytics(result)

#     annotated_frame = bounding_box_annotator.annotate(
#         scene=frame.copy(), detections=detections)

#     sv.plot_image(image=annotated_frame, size=(8, 8))

import os
import math

import supervision as sv
from ultralytics import YOLO
import cv2

import numpy as np
import matplotlib.pyplot as plt


def load_model(model_path):
    model = YOLO(model_path)
    return model

def get_frame_generator(video_path):
    frame_generator = sv.get_video_frames_generator(video_path)
    return frame_generator


def display_annotated_frame(frame, result):
    detections = sv.Detections.from_ultralytics(result)
    # Create a bounding box annotator
    bounding_box_annotator = sv.BoundingBoxAnnotator()

    # Annotate the image with bounding boxes
    annotated_frame = bounding_box_annotator.annotate(scene=frame.copy(), detections=detections)

    # Display the annotated frame
    cv2.imshow('Detected Objects', annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_frames(frame_generator, model, classNames, label, threshold=0.2, show_bounding_box=True):
    trajectory = []
    frame_count = 0
    for frame in frame_generator:
        # Perform inference on the current frame
        results = model(frame, stream=False)
        
        if results is not None:  # Check if results is not None
            for r in results:
                boxes = r.boxes

                for box in boxes:
                    # Bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Confidence
                    confidence = math.ceil((box.conf[0] * 100)) / 100
                    print("Confidence --->", confidence)

                    # Class name
                    cls = int(box.cls[0])
                    
                    print("Class name -->", classNames[cls])

                    # Store coordinates of the bowl's trajectory
                    if cls == 0:  # Assuming bowl is class 0
                        current_x = x1 + (x2 - x1) // 2
                        trajectory.append((current_x, frame_count, y1 + (y2 - y1) // 2))  # Store x, y, z
                        print(f"Our frame {frame_count}.")
            
        else:
            print(f"No results detected for frame {frame_count}.")
        
        frame_count += 1
    return trajectory
       

def convert_to_numpy_array(trajectory):
    return np.array(trajectory)

def filter_trajectory(trajectory, max_x_deviation):
    filtered_trajectory = []
    for i in range(1, len(trajectory)):
        if (trajectory[i][0] - trajectory[i-1][0]) > 0:
            print(trajectory[i][0], " - ", trajectory[i-1][0], " = ", trajectory[i][0] - trajectory[i-1][0])
            filtered_trajectory.append(trajectory[i])
    return np.array(filtered_trajectory)

def reverse_z_coordinates(trajectory):
    # Reverse the order of z-coordinates
    return np.array([(x, y, np.max(trajectory[:, 2]) - z) for x, y, z in trajectory])


def plot_trajectory_2d(trajectory):
    plt.plot(trajectory[:, 0], trajectory[:, 2], 'r-', label='Filtered Trajectory')
    plt.scatter(trajectory[0, 0], trajectory[0, 2], c='g', marker='o', label='Throwing Point')
    plt.scatter(trajectory[-1, 0], trajectory[-1, 2], c='b', marker='o', label='Destination Point')
    plt.xlabel('X-coordinate')
    plt.ylabel('Height')  # Z-coordinate representing height
    plt.title('Filtered Trajectory of the Bowl in 2D')
    plt.legend()
    plt.show()



def process_video(video_path, model_path, classNames = ["bowl", "bowler"], chunk_size=50, label='bowl', threshold=0.5):
    # Load model
    model = load_model(model_path)
    
    # Get video frame generator
    frame_generator = get_frame_generator(video_path)      
    # Define trajectory and process frames in chunks
    trajectory = process_frames(frame_generator, model, classNames, chunk_size, label, threshold)  # Correct order of arguments

    # Post-process the trajectory
    trajectory = post_process_trajectory(trajectory)
    
    print(trajectory)

    # Plot the filtered trajectory in 2D
    plot_trajectory_2d(trajectory)


def post_process_trajectory(trajectory):
    # Convert trajectory to numpy array
    trajectory = convert_to_numpy_array(trajectory)

    # Define constants for constraints
    MAX_X_DEVIATION = 5  # Maximum deviation in X-coordinate from previous point

    # Filter and reverse trajectory
    trajectory = filter_trajectory(trajectory, MAX_X_DEVIATION)
    trajectory = reverse_z_coordinates(trajectory)
    
    return trajectory



