import cv2
from ultralytics import YOLO
import time
from collections import deque

model = YOLO("yolov8x-seg.pt")

cap = cv2.VideoCapture(0)
fps_history = deque(maxlen=30)

while cap.isOpened():
    t_start = time.perf_counter()
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)
    results = model(frame)
    annotated_frame = results[0].plot()

    t_stop = time.perf_counter()
    frame_time = t_stop - t_start
    fps_history.append(1 / frame_time if frame_time > 0 else 0)
    avg_fps = sum(fps_history) / len(fps_history)

    cv2.putText(
        annotated_frame,
        f"FPS: {avg_fps:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    cv2.imshow("YOLO Segmentation", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
