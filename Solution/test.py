from ultralytics import YOLO

model = YOLO('best.pt')

results = model.predict('D:/Assignment/dataset/images/test/344_png_jpg.rf.a30885c95ef92a1b8ed8cd5819232eda.jpg', conf=0.25)
results[0].show()


