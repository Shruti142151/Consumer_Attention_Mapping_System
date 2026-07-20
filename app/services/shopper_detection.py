import cv2
from ultralytics import YOLO
from collections import defaultdict
from session_tracking import SessionTracker


def start_shopper_detection():
    # Load YOLOv8 model
    model = YOLO("yolov8n.pt")
    shopper_paths = defaultdict(list)
    session_tracker = SessionTracker()

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera could not be opened")
        return

    print("Shopper detection started. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Run YOLO detection
        results = model.track(
            frame,
            persist=True,
            classes=[0]
        )

        # Draw detections
        for result in results:
            boxes = result.boxes

            for box in boxes:
                # Get class ID
                cls = int(box.cls[0])

                # Detect only people (class 0 in COCO dataset)
                if cls == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    confidence = float(box.conf[0])

                    track_id = box.id

                    if track_id is not None:
                        track_id = int(track_id[0])

                        session_tracker.shopper_entered(track_id)
                        session_tracker.shopper_seen(track_id)

                        center_x = int((x1+x2)/2)
                        center_y = int((y1+y2)/2)

                        shopper_paths[track_id].append((center_x, center_y))



                    # Draw bounding box
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Shopper ID: {track_id} {confidence:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

                    for point in shopper_paths[track_id]:
                        cv2.circle(
                            frame,
                            point,
                            3,
                            (255, 0, 0),
                            -1
                        )

        # Display output
        cv2.imshow("Shopper Detection Engine", frame)

        # Press q to close
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_shopper_detection()