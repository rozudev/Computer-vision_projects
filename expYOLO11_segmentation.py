from ultralytics import YOLOE

model = YOLOE("yolo11n-seg.pt")

results = model(source=0, imgsz=640, conf=0.35, stream=True, show=True)
