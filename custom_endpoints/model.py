import base64
import io
import logging
import os
import uuid
from typing import Any

from PIL import Image
from ultralytics import YOLO

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Model:
    def __init__(self, **kwargs) -> None:
        self._model_name = "yolo"
        self._data_dir = kwargs.get("data_dir", None)
        model_metadata = kwargs.get("config", {}).get("model_metadata", {})
        self._model_binary_dir = model_metadata.get("model_binary_dir", "")

        self.detection_model = None
        self.segmentation_model = None

        logger.info(f"YOLO Model initialized with data_dir: {self._data_dir}")

    def load(self) -> None:
        base_path = os.path.join(str(self._data_dir), str(self._model_binary_dir))
        detection_model_filepath = os.path.join(base_path, "yolo26n.pt")
        segmentation_model_filepath = os.path.join(base_path, "yolo26n-seg.pt")

        if not os.path.exists(detection_model_filepath):
            raise FileNotFoundError(
                f"Detection model file {detection_model_filepath} not found"
            )

        if not os.path.exists(segmentation_model_filepath):
            raise FileNotFoundError(
                f"Segmentation model file {segmentation_model_filepath} not found"
            )

        logger.info(f"Loading detection model file {detection_model_filepath}")
        self.detection_model = YOLO(detection_model_filepath)

        logger.info(f"Loading segmentation model file {segmentation_model_filepath}")
        self.segmentation_model = YOLO(segmentation_model_filepath)

        logger.info("YOLO model loaded")

    def get_image_from_input(self, model_input: Any) -> Image.Image:
        """Convert user input image from base64 to PIL Image.

        User input is expected to be a JSON object with a key "image_base64"
        containing the base64 encoded image.
        """

        image_base64 = model_input.get("image_base64", "")
        if not image_base64:
            raise ValueError("image_base64 is required")

        return Image.open(io.BytesIO(base64.b64decode(image_base64)))

    def get_models(self) -> dict:
        """Get the available models."""

        return {
            "models": [
                {
                    "name": "yolo26n",
                    "checkpoint": self.detection_model.ckpt_path,
                    "date": self.detection_model.ckpt["date"],
                    "task": self.detection_model.task,
                },
                {
                    "name": "yolo26n-seg",
                    "checkpoint": self.segmentation_model.ckpt_path,
                    "date": self.segmentation_model.ckpt["date"],
                    "task": self.segmentation_model.task,
                },
            ]
        }

    def detect_image(self, model_input: Any) -> dict:
        """Detect objects in the image.

        Returns the bounding boxes, confidence, and label.
        """

        image = self.get_image_from_input(model_input)

        # Do inference.
        output = self.detection_model(image)
        result = output[0]

        # Format the results.
        boxes = result.boxes
        predictions = []
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                predictions.append(
                    {
                        "bbox": box.xyxy.tolist(),
                        "confidence": box.conf.tolist(),
                        "label": result.names[int(box.cls.item())],
                    }
                )

        return {"predictions": predictions}

    def segment_image(self, model_input: Any) -> dict:
        """Segment the image and return the image as base64."""

        image = self.get_image_from_input(model_input)

        # Do inference.
        output = self.segmentation_model(image)
        result = output[0]

        # Convert the segmentation plot (tensor) to a WEBP image.
        seg_image = Image.fromarray(result.plot())
        unique_name = f"/tmp/seg_image_{uuid.uuid4().hex}.webp"
        seg_image.save(unique_name)
        with open(unique_name, "rb") as image_file:
            return_image = base64.b64encode(image_file.read()).decode("utf-8")

        return {"segmented_image": return_image}
