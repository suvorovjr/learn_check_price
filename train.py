from ultralytics import YOLO

model = YOLO("yolov8l.pt")


model.train(
    data="dataset.yaml",
    epochs=100,
    imgsz=800,
    batch=16,
    device="cpu",
    project="runs",
    name="price_tag_yolov8l"
)

