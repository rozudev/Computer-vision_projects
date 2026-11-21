from ultralytics import FastSAM

# Lightweight CNN-based SAM (~23 MB) — orders of magnitude faster on CPU
model = FastSAM("FastSAM-s.pt")

# stream=True prevents RAM from filling up on live video
results = model(source=0, imgsz=640, conf=0.4, stream=True, show=True, save=False)
