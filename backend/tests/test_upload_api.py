import base64

from fastapi.testclient import TestClient

from main_server import app


def test_analyze_image_accepts_uploaded_file():
    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB4Lq6AAAAAXNSR0IArs4c6QAAAC0lEQVR42mP8z8AARQAB6wN9" \
        "dwAABQABJRU5ErkJggg=="
    )

    client = TestClient(app)
    response = client.post(
        "/api/analyze-image",
        files={"file": ("test.png", png_bytes, "image/png")},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["source"] == "uploaded_file"
    assert payload["filename"] == "test.png"
    assert "processed_image_base64" in payload
    assert payload["preprocessing"]["filter"]
