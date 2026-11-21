from ultralytics import YOLO

model = YOLO("yolo11n-pose.pt")

results = model(source=0, conf=0.25, show=True, save=True)