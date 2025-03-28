from ultralytics import YOLO

model = YOLO('models/best.pt')

results = model.predict("C:/Users/Harsh/Documents/GitHub/Football-Analysis-system-using-yolo-opencv-python/input_video/video.mp4",save=True)
print(results[0])

for box in results[0].boxes:
    print(box)
