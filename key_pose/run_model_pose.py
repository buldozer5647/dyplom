import cv2
import time
from collections import deque
from ultralytics import YOLO

model = YOLO("best.pt")
cap = cv2.VideoCapture(0)

fps_history = deque(maxlen=30)

while cap.isOpened():
    start_time = time.perf_counter()
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()

    hand_count = 0
    for kp in results[0].keypoints.data:
        visible = kp[:, 2] > 0.5
        if visible.any():
            hand_count += 1

    end_time = time.perf_counter()
    current_fps = 1 / (end_time - start_time)
    fps_history.append(current_fps)
    avg_fps = sum(fps_history) / len(fps_history)

    cv2.putText(annotated_frame, f"FPS: {avg_fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Hands: {hand_count}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Pose Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
