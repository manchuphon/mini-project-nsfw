"""
test_api.py – Unit tests for NSFW Image Classification API

Covers:
  - GET /health
  - POST /predict  (happy path + edge cases)
  - POST /predict/batch
"""
import io
import pytest
from PIL import Image


# ===========================================================================
# 1. Health Check
# ===========================================================================
class TestHealth:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_json_structure(self, client):
        data = response = client.get("/health").json()
        assert "status" in data
        assert data["status"] == "ok"


# ===========================================================================
# 2. POST /predict – Happy Path
# ===========================================================================
class TestPredictHappyPath:
    def test_predict_png_returns_200(self, client, valid_png_file):
        name, data, ctype = valid_png_file
        response = client.post(
            "/predict",
            files={"file": (name, data, ctype)},
        )
        assert response.status_code == 200

    def test_predict_response_has_required_fields(self, client, valid_png_file):
        name, data, ctype = valid_png_file
        body = client.post(
            "/predict",
            files={"file": (name, data, ctype)},
        ).json()

        for field in ("filename", "label", "confidence", "scores", "latency_ms"):
            assert field in body, f"Missing field: {field}"

    def test_predict_label_is_valid(self, client, valid_png_file):
        name, data, ctype = valid_png_file
        body = client.post(
            "/predict",
            files={"file": (name, data, ctype)},
        ).json()
        assert body["label"] in ("normal", "nsfw")

    def test_predict_confidence_in_range(self, client, valid_png_file):
        name, data, ctype = valid_png_file
        body = client.post(
            "/predict",
            files={"file": (name, data, ctype)},
        ).json()
        assert 0.0 <= body["confidence"] <= 1.0

    def test_predict_scores_sum_to_one(self, client, valid_png_file):
        name, data, ctype = valid_png_file
        body = client.post(
            "/predict",
            files={"file": (name, data, ctype)},
        ).json()
        total = sum(body["scores"].values())
        assert abs(total - 1.0) < 1e-3, f"Scores don't sum to 1: {total}"

    def test_predict_scores_keys(self, client, valid_png_file):
        name, data, ctype = valid_png_file
        body = client.post(
            "/predict",
            files={"file": (name, data, ctype)},
        ).json()
        assert set(body["scores"].keys()) == {"normal", "nsfw"}

    def test_predict_filename_echoed(self, client, valid_png_file):
        name, data, ctype = valid_png_file
        body = client.post(
            "/predict",
            files={"file": (name, data, ctype)},
        ).json()
        assert body["filename"] == name

    def test_predict_latency_is_positive(self, client, valid_png_file):
        name, data, ctype = valid_png_file
        body = client.post(
            "/predict",
            files={"file": (name, data, ctype)},
        ).json()
        assert body["latency_ms"] > 0

    def test_predict_jpeg_works(self, client, valid_jpeg_file):
        name, data, ctype = valid_jpeg_file
        response = client.post(
            "/predict",
            files={"file": (name, data, ctype)},
        )
        assert response.status_code == 200
        assert response.json()["label"] in ("normal", "nsfw")

    def test_predict_webp_works(self, client):
        """WebP should also be accepted."""
        buf = io.BytesIO()
        Image.new("RGB", (64, 64), (100, 150, 200)).save(buf, format="WEBP")
        buf.seek(0)
        response = client.post(
            "/predict",
            files={"file": ("photo.webp", buf.read(), "image/webp")},
        )
        assert response.status_code == 200


# ===========================================================================
# 3. POST /predict – Error Handling
# ===========================================================================
class TestPredictErrors:
    def test_unsupported_content_type_returns_415(self, client, text_file_as_image):
        name, data, ctype = text_file_as_image
        response = client.post(
            "/predict",
            files={"file": (name, data, ctype)},
        )
        assert response.status_code == 415

    def test_empty_file_returns_400(self, client, empty_file):
        name, data, ctype = empty_file
        response = client.post(
            "/predict",
            files={"file": (name, data, ctype)},
        )
        assert response.status_code == 400

    def test_corrupted_image_returns_400(self, client, corrupted_file):
        name, data, ctype = corrupted_file
        response = client.post(
            "/predict",
            files={"file": (name, data, ctype)},
        )
        assert response.status_code == 400

    def test_oversized_file_returns_413(self, client, oversized_file):
        name, data, ctype = oversized_file
        response = client.post(
            "/predict",
            files={"file": (name, data, ctype)},
        )
        assert response.status_code == 413

    def test_error_response_has_detail_field(self, client, empty_file):
        name, data, ctype = empty_file
        body = client.post(
            "/predict",
            files={"file": (name, data, ctype)},
        ).json()
        assert "detail" in body

    def test_no_file_returns_422(self, client):
        """Missing required field should return 422 Unprocessable Entity."""
        response = client.post("/predict")
        assert response.status_code == 422


# ===========================================================================
# 4. POST /predict/batch
# ===========================================================================
class TestPredictBatch:
    def _make_files(self, count: int):
        files = []
        for i in range(count):
            buf = io.BytesIO()
            Image.new("RGB", (64, 64), (i * 10 % 256, 100, 200)).save(buf, format="PNG")
            buf.seek(0)
            files.append(("files", (f"img_{i}.png", buf.read(), "image/png")))
        return files

    def test_batch_single_image(self, client):
        files = self._make_files(1)
        response = client.post("/predict/batch", files=files)
        assert response.status_code == 200

    def test_batch_response_structure(self, client):
        files = self._make_files(3)
        body = client.post("/predict/batch", files=files).json()
        assert "results" in body
        assert "count" in body
        assert body["count"] == 3
        assert len(body["results"]) == 3

    def test_batch_each_result_has_label(self, client):
        files = self._make_files(2)
        body = client.post("/predict/batch", files=files).json()
        for r in body["results"]:
            assert "label" in r or "error" in r

    def test_batch_max_16_images(self, client):
        files = self._make_files(16)
        response = client.post("/predict/batch", files=files)
        assert response.status_code == 200

    def test_batch_over_limit_returns_400(self, client):
        files = self._make_files(17)
        response = client.post("/predict/batch", files=files)
        assert response.status_code == 400

    def test_batch_mixed_valid_invalid(self, client):
        """Batch with one bad file should not crash the entire request."""
        buf = io.BytesIO()
        Image.new("RGB", (64, 64)).save(buf, format="PNG")
        buf.seek(0)
        files = [
            ("files", ("good.png", buf.read(), "image/png")),
            ("files", ("bad.txt", b"not an image", "text/plain")),
        ]
        response = client.post("/predict/batch", files=files)
        # Should still return 200 – bad file gets an error entry in results
        assert response.status_code == 200
        body = response.json()
        errors = [r for r in body["results"] if "error" in r]
        assert len(errors) >= 1


# ===========================================================================
# 5. Model Prediction Sanity Check
# ===========================================================================
class TestModelSanity:
    """
    Verify the model itself returns meaningful predictions
    (not always the same class, confidence is a float, etc.)
    """

    def test_model_returns_consistent_label_for_same_input(self, client, valid_png_file):
        name, data, ctype = valid_png_file
        results = set()
        for _ in range(3):
            body = client.post(
                "/predict",
                files={"file": (name, data, ctype)},
            ).json()
            results.add(body["label"])
        # Same image → same label every time (deterministic model)
        assert len(results) == 1, "Model should be deterministic for the same input"

    def test_model_confidence_is_float(self, client, valid_png_file):
        name, data, ctype = valid_png_file
        body = client.post(
            "/predict",
            files={"file": (name, data, ctype)},
        ).json()
        assert isinstance(body["confidence"], float)

    def test_model_label_matches_highest_score(self, client, valid_png_file):
        name, data, ctype = valid_png_file
        body = client.post(
            "/predict",
            files={"file": (name, data, ctype)},
        ).json()
        best = max(body["scores"], key=lambda k: body["scores"][k])
        assert body["label"] == best, "label should match the class with highest score"