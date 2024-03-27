
from src.utils import CalculateSpeed
from src.trajectory import process_video

video_path = './test_videos/1.mp4'
# model_path = './models/best.pt'

# obj_detect = CalculateSpeed(video_path=video_path, model_path=model_path)

# obj_detect.calculate_speed()

model_path = "./models/initial_best.pt"
process_video(video_path, model_path)