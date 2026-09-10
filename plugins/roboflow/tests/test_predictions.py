from vision_agents.plugins.roboflow.predictions import (
    detection_state,
    detections_from_predictions,
)


def _prediction(label: str, class_id: int, x: int, y: int, w: int, h: int, conf: float):
    return {
        "class": label,
        "class_id": class_id,
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "confidence": conf,
    }


class TestDetectionsFromPredictions:
    def test_empty_payload(self):
        detections, labels, objects = detections_from_predictions(None)
        assert objects == []
        assert labels == {}
        assert len(detections) == 0

    def test_inference_response(self):
        payload = {
            "predictions": [
                _prediction("rose", 0, 50, 40, 20, 10, 0.9),
                _prediction("tulip", 1, 100, 80, 40, 20, 0.8),
            ]
        }

        detections, labels, objects = detections_from_predictions(payload)
        assert labels == {0: "rose", 1: "tulip"}
        assert objects[0] == {"label": "rose", "x1": 40, "y1": 35, "x2": 60, "y2": 45}
        assert objects[1]["label"] == "tulip"
        assert len(detections) == 2

    def test_filters_classes(self):
        payload = {
            "predictions": [
                _prediction("rose", 0, 10, 10, 4, 4, 0.9),
                _prediction("tulip", 1, 20, 20, 4, 4, 0.8),
            ]
        }
        _, _, objects = detections_from_predictions(payload, classes=["rose"])
        assert [obj["label"] for obj in objects] == ["rose"]

    def test_nested_workflow_payload(self):
        payload = {
            "predictions": {"predictions": [_prediction("daisy", 2, 8, 8, 4, 4, 0.7)]}
        }
        _, _, objects = detections_from_predictions(payload)
        assert objects[0]["label"] == "daisy"


class TestDetectionState:
    def test_counts_labels(self):
        objects = [
            {"label": "rose", "x1": 0, "y1": 0, "x2": 1, "y2": 1},
            {"label": "rose", "x1": 2, "y1": 2, "x2": 3, "y2": 3},
            {"label": "tulip", "x1": 4, "y1": 4, "x2": 5, "y2": 5},
        ]
        snapshot = detection_state(objects, confidences=[0.9, 0.8, 0.7])
        assert snapshot["counts"] == {"rose": 2, "tulip": 1}
        assert snapshot["objects"][0]["label"] == "rose"
        assert snapshot["objects"][0]["confidence"] == 0.9
        assert isinstance(snapshot["updated_at"], float)

    def test_empty(self):
        assert detection_state([])["objects"] == []
        assert detection_state([])["counts"] == {}
