"""
YOLOv11 vehicle detector with built-in ByteTrack tracking.

Supports both COCO-pretrained and VisDrone-finetuned models.
VisDrone models are required for drone/aerial parking footage.
"""

from pathlib import Path

from tqdm import tqdm
from ultralytics import YOLO

from src.detection.base import DetectedVehicle, FrameDetections, ParkingDetector

# COCO class IDs for vehicles
COCO_VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# VisDrone class IDs for vehicles
VISDRONE_VEHICLE_CLASSES = {3: "car", 4: "van", 5: "truck", 8: "bus"}

# DLP class IDs (from prepare_dlp_dataset.py)
DLP_VEHICLE_CLASSES = {0: "car", 1: "medium vehicle", 2: "bus"}

VISDRONE_PERSON_CLASSES = {0: "pedestrian", 1: "people"}
VISDRONE_ALL_CLASSES = {**VISDRONE_PERSON_CLASSES, **VISDRONE_VEHICLE_CLASSES}

def _detect_class_map(model: YOLO) -> dict[int, str]:
    """Auto-detect whether the model uses COCO, VisDrone, or DLP classes."""
    names = model.names
    # DLP fine-tuned: 3 classes starting with Car
    if names.get(0) == "Car" and names.get(1) == "Medium Vehicle":
        return DLP_VEHICLE_CLASSES
    # VisDrone: car at index 3, van at index 4
    if names.get(3) == "car" and names.get(4) == "van":
        return VISDRONE_ALL_CLASSES # ← include pedestrians for evaluation
    return COCO_VEHICLE_CLASSES


class YOLODetector(ParkingDetector):
    """
    YOLOv11-based vehicle detector with ByteTrack tracking.

    Auto-detects whether the loaded model uses COCO or VisDrone class IDs.
    """

    def __init__(self, model_name: str = "yolo11n.pt", conf_threshold: float = 0.25):
        super().__init__()
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        self.vehicle_classes = _detect_class_map(self.model)
        class_source = "VisDrone" if self.vehicle_classes is VISDRONE_VEHICLE_CLASSES else "COCO"
        print(f"Loaded {model_name} with {class_source} classes")
        print(f"classes: {self.vehicle_classes}")

    def detect_and_track(
        self,
        video_path: str,
        tracker: str = "bytetrack.yaml",
        imgsz: int = 1920,
        stream: bool = True,
        **kwargs,
    ) -> list[FrameDetections]:
        """
        Run YOLOv11 detection + ByteTrack tracking on a video.

        Args:
            video_path: Path to the video file.
            tracker: Tracker config (default ByteTrack).
            imgsz: Inference image size (width). 1920 balances speed and accuracy for 4K.
            stream: Stream results to reduce memory usage.

        Returns:
            List of FrameDetections with tracked vehicles.
        """
        video_path = str(Path(video_path).resolve())

        results = self.model.track(
            source=video_path,
            tracker=tracker,
            stream=stream,
            persist=True,
            imgsz=imgsz,
            conf=self.conf_threshold,
            classes=list(self.vehicle_classes.keys()),
            verbose=False,
        )

        all_detections = []
        fps_val = None

        for frame_idx, result in enumerate(tqdm(results, desc="Processing video")):
            if fps_val is None:
                try:
                    fps_val = self.model.predictor.dataset.cap.get(5)  # cv2.CAP_PROP_FPS
                except Exception:
                    fps_val = 25.0

            timestamp = frame_idx / fps_val

            vehicles = []
            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].item())
                    if cls_id not in self.vehicle_classes:
                        continue

                    track_id = int(boxes.id[i].item())
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                    conf = float(boxes.conf[i].item())
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2

                    vehicles.append(
                        DetectedVehicle(
                            track_id=track_id,
                            bbox=(x1, y1, x2, y2),
                            confidence=conf,
                            class_name=self.vehicle_classes[cls_id],
                            center_px=(cx, cy),
                        )
                    )

            all_detections.append(
                FrameDetections(
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                    vehicles=vehicles,
                )
            )

        return all_detections
