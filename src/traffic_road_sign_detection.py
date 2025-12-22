import cv2 # type: ignore
import csv
from ultralytics import YOLO # type: ignore

model = YOLO("C:\\Users\\bavan\\Downloads\\bosch-traffic-sign\\yolov8n.pt")  

input_video = "C:\\Users\\bavan\\Downloads\\bosch-traffic-sign\\traffic-sign-input.mp4"
cap = cv2.VideoCapture(input_video)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_video = "output_traffic_signs.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

csv_file = open("traffic_sign_predictions.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "timestamp_sec",
    "frame_id",
    "class_id",
    "class_name",
    "confidence",
    "x1", "y1", "x2", "y2"
])

frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    timestamp = frame_id / fps 
    results = model(frame, conf=0.4)
    detections = results[0]

    for box in detections.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = model.names[cls_id]

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        csv_writer.writerow([
            round(timestamp, 3),
            frame_id,
            cls_id,
            class_name,
            round(conf, 3),
            x1, y1, x2, y2
        ])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{class_name} {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    out.write(frame)

    cv2.imshow("Traffic & Road Sign Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_id += 1

cap.release()
out.release()
csv_file.close()
cv2.destroyAllWindows()

print("Inference complete!")
print("Saved video:", output_video)
print("Saved logs: traffic_sign_predictions.csv")
