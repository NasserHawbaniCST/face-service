# -*- coding: utf-8 -*-
"""The REAL OpenCV models (YuNet + SFace) through the API.

No real face photos exist in the repo, so this proves: the model files load and
match the published hashes, blank / noisy pictures yield no face, the decode
guards refuse absurd sizes BEFORE decoding, large images are downscaled, and the
recogniser produces a 128-d vector. Real-world accuracy is NOT covered here."""
import base64
import hashlib
import os
import struct

import pytest

cv2 = pytest.importorskip('cv2')
np = pytest.importorskip('numpy')

from app.engine import (  # noqa: E402
    DEFAULT_MODELS_DIR, DETECTOR_FILE, EXPECTED_SHA256, MAX_ANALYSIS_EDGE, RECOGNIZER_FILE, EngineUnavailable,
    Face, OpenCVEngine, jpeg_dimensions, landmark_yaw,
)
from app.main import opencv_factory  # noqa: E402
from app.config import load_settings  # noqa: E402
from conftest import Signer, make_env  # noqa: E402

URL = '/v1/analyze'


def jpeg(img):
    ok, buf = cv2.imencode('.jpg', img)
    assert ok
    return bytes(buf)


def b64(data):
    return base64.b64encode(data).decode()


@pytest.fixture(scope='module')
def engine():
    return OpenCVEngine()


@pytest.fixture
def real_client(make_client):
    return make_client(engine_factory=opencv_factory(load_settings(make_env())), FACE_WORKERS=1)


def post(client, frames, request_id='real-1'):
    return client.post(URL, **Signer().request({'request_id': request_id, 'frames': frames}))


def test_model_files_are_the_published_ones():
    for name, expected in EXPECTED_SHA256.items():
        with open(os.path.join(DEFAULT_MODELS_DIR, name), 'rb') as handle:
            assert hashlib.sha256(handle.read()).hexdigest() == expected


def test_a_tampered_model_is_refused(tmp_path):
    for name in (DETECTOR_FILE, RECOGNIZER_FILE):
        with open(os.path.join(DEFAULT_MODELS_DIR, name), 'rb') as src:
            data = src.read() if name == DETECTOR_FILE else b'not the real model'
        (tmp_path / name).write_bytes(data)
    with pytest.raises(EngineUnavailable):
        OpenCVEngine(models_dir=str(tmp_path))
    with pytest.raises(EngineUnavailable):
        OpenCVEngine(models_dir=str(tmp_path / 'missing'))


def test_health_lists_the_loaded_hashes(real_client):
    body = real_client.get('/health').json()
    assert body['status'] == 'ok'
    assert body['models']['detector']['sha256'] == EXPECTED_SHA256[DETECTOR_FILE]
    assert body['models']['recognizer']['sha256'] == EXPECTED_SHA256[RECOGNIZER_FILE]


def test_blank_image_has_no_face(real_client):
    blank = np.full((480, 360, 3), 127, dtype=np.uint8)
    resp = post(real_client, [{'id': 'blank', 'image_b64': b64(jpeg(blank))}])
    assert resp.status_code == 200, resp.text
    f = resp.json()['frames'][0]
    assert f['ok'] is True and f['faces'] == 0
    assert (f['width'], f['height']) == (360, 480)
    assert f['box'] is None and f['landmarks'] is None and f['yaw'] is None
    assert f['embedding'] is None


def test_noise_image_has_no_face(engine):
    rng = np.random.default_rng(1)
    noise = rng.integers(0, 255, (480, 360, 3), dtype=np.uint8)
    result = engine.analyze_jpeg(jpeg(noise))
    assert result['ok'] and result['faces'] == 0


def test_big_image_is_downscaled(engine):
    result = engine.analyze_jpeg(jpeg(np.full((2000, 1500, 3), 100, dtype=np.uint8)))
    assert max(result['width'], result['height']) == MAX_ANALYSIS_EDGE
    assert (result['width'], result['height']) == (480, 640)


def test_absurd_dimensions_refused_before_decoding(engine, monkeypatch):
    # a tiny file that CLAIMS to be 30000 x 30000 px
    bomb = (b'\xff\xd8\xff\xc0\x00\x11\x08' + struct.pack('>HH', 30000, 30000)
            + b'\x03\x01\x11\x00\x02\x11\x01\x03\x11\x01' + b'\xff\xd9')
    assert jpeg_dimensions(bomb) == (30000, 30000)

    def must_not_decode(*a, **k):
        raise AssertionError('cv2.imdecode was called on a decompression bomb')
    monkeypatch.setattr(cv2, 'imdecode', must_not_decode)
    result = engine.analyze_jpeg(bomb)
    assert (result['ok'], result['error']) == (False, 'bad_image')


def test_non_jpeg_and_garbage_are_bad_images(engine):
    for data in (b'', b'GIF89a not a jpeg', b'\xff\xd8\xff garbage', b'\x89PNG\r\n\x1a\n' + b'0' * 50):
        result = engine.analyze_jpeg(data)
        assert (result['ok'], result['error']) == (False, 'bad_image'), data[:8]


def test_feature_extractor_gives_128_numbers(engine):
    rng = np.random.default_rng(2)
    img = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    row = np.array([100, 80, 100, 120, 130, 110, 190, 110, 160, 150, 135, 180, 190, 175, 0.99], dtype=np.float32)
    face = Face(box=(100, 80, 100, 120), landmarks=[(130, 110), (190, 110), (160, 150)], score=0.99, raw=row)
    vec = engine.embed(img, face)
    assert len(vec) == 128
    assert 0.0 <= engine.sharpness(img, face) <= 1.0


def test_sharpness_distinguishes_blur(engine):
    rng = np.random.default_rng(3)
    sharp = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    blurred = cv2.GaussianBlur(sharp, (31, 31), 0)
    face = Face(box=(80, 60, 120, 140))
    assert engine.sharpness(sharp, face) > engine.sharpness(blurred, face)


def test_landmark_yaw_formula():
    frontal = Face(box=(0, 0, 100, 100), landmarks=[(30, 40), (70, 40), (50, 60), (35, 80), (65, 80)])
    right = Face(box=(0, 0, 100, 100), landmarks=[(30, 40), (70, 40), (62, 60), (35, 80), (65, 80)])
    left = Face(box=(0, 0, 100, 100), landmarks=[(30, 40), (70, 40), (38, 60), (35, 80), (65, 80)])
    assert landmark_yaw(frontal) == pytest.approx(0.0)
    assert landmark_yaw(right) == pytest.approx(0.3)
    assert landmark_yaw(left) == pytest.approx(-0.3)
    assert landmark_yaw(Face(box=(0, 0, 1, 1))) == 0.0
