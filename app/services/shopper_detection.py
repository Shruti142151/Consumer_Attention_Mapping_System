import cv2
from ultralytics import YOLO
from collections import defaultdict

from attention_analysis import AttentionTracker
from session_tracking import SessionTracker


# Load YOLO model
model = YOLO("yolov8n.pt")


# Trackers
shopper_paths = defaultdict(list)

session_tracker = SessionTracker()
attention_tracker = AttentionTracker()


# OpenCV window setup
cv2.namedWindow("Shopper Detection Engine", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Shopper Detection Engine", 1280, 720)


# Open webcam
cap = cv2.VideoCapture(0)


while True:

    ret, frame = cap.read()

    if not ret:
        break


    # YOLO tracking
    results = model.track(
        frame,
        persist=True,
        classes=[0]
    )


    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy()


        for box, track_id in zip(boxes, track_ids):

            track_id = int(track_id)


            # Attention tracking
            attention_tracker.start_attention(track_id)

            duration = attention_tracker.get_attention_duration(track_id)


            # Session tracking
            session_tracker.shopper_entered(track_id)
            session_tracker.shopper_seen(track_id)


            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, box)


            # Center point for path tracking
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)


            shopper_paths[track_id].append(
                (center_x, center_y)
            )


            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # Draw shopper ID
            cv2.putText(
                frame,
                f"Shopper ID: {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


            # Draw attention duration
            cv2.putText(
                frame,
                f"Attention: {duration:.1f}s",
                (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )


            # Draw path
            points = shopper_paths[track_id]

            for i in range(1, len(points)):

                cv2.line(
                    frame,
                    points[i - 1],
                    points[i],
                    (255, 0, 0),
                    2
                )


    # Display output
    cv2.imshow(
        "Shopper Detection Engine",
        frame
    )


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break



cap.release()
cv2.destroyAllWindows()