# -*- coding: utf-8 -*-
"""Face engine of the service: YuNet (detector) + SFace (recogniser) via OpenCV.

THIS FILE MIRRORS `ai_addons/cs_identity_gateway/services/face_engine.py`
(OpenCVFaceEngine) on purpose: same model files, same decode guards, same
downscale rule, same landmark / yaw formula, same box / area semantics. The
gateway addon has a parity test (tests/test_remote.py) that loads this file and
compares the two on synthetic inputs - if you change the logic here, change it
there too (and bump MODEL_VERSION only when the MODELS change).

Self-contained on purpose (stdlib + cv2 + numpy only, no web framework import)
so that test can load it.

Everything works on RAM only: JPEG bytes in, numbers out. Nothing is written to
disk, nothing is logged.
"""
import hashlib
import math
import os
import struct
import threading
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

DETECTOR_FILE = 'face_detection_yunet_2023mar.onnx'
RECOGNIZER_FILE = 'face_recognition_sface_2021dec.onnx'
MODEL_VERSION = 'yunet_2023mar+sface_2021dec'      # same string as the gateway addon

# sha256 of the unmodified OpenCV Zoo files (MIT / Apache-2.0)
EXPECTED_SHA256 = {
    DETECTOR_FILE: '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4',
    RECOGNIZER_FILE: '0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79',
}

# Decompression-bomb guard: refuse BEFORE decoding (read from the JPEG header).
MAX_IMAGE_PIXELS = 8_000_000
# Frames are analysed at most at this long edge (the app sends <= 480 px).
MAX_ANALYSIS_EDGE = 640

DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')

# frame-level error codes
ERR_BAD_IMAGE = 'bad_image'


class EngineUnavailable(Exception):
    """cv2 / numpy / the model files are missing or do not match."""


@dataclass
class Face:
    """One detected face. `landmarks` = 5 points (x, y): right eye, left eye,
    nose tip, right mouth corner, left mouth corner (image coordinates)."""
    box: Tuple[float, float, float, float]            # x, y, w, h
    landmarks: Sequence[Tuple[float, float]] = field(default_factory=list)
    score: float = 1.0
    raw: Any = None                                    # engine private payload

    @property
    def area(self):
        return max(self.box[2], 0.0) * max(self.box[3], 0.0)


def landmark_yaw(face):
    """Nose vs eye-midpoint offset divided by the inter-eye distance."""
    if not face.landmarks or len(face.landmarks) < 3:
        return 0.0
    (rx, ry), (lx, ly), (nx, ny) = face.landmarks[0], face.landmarks[1], face.landmarks[2]
    eye_dist = math.hypot(lx - rx, ly - ry)
    if eye_dist < 1e-6:
        return 0.0
    return ((nx - (rx + lx) / 2.0)) / eye_dist


def jpeg_dimensions(data):
    """(width, height) read from the JPEG header without decoding it, or None
    when `data` is not a JPEG."""
    if len(data) < 4 or data[:2] != b'\xff\xd8':
        return None
    i = 2
    n = len(data)
    while i + 4 <= n:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker == 0xFF:
            i += 1
            continue
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if marker == 0xD9:
            return None
        if i + 4 > n:
            return None
        (seg_len,) = struct.unpack('>H', data[i + 2:i + 4])
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            if i + 9 > n:
                return None
            height, width = struct.unpack('>HH', data[i + 5:i + 9])
            return width, height
        i += 2 + seg_len
    return None


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def r(value, places):
    return round(float(value), places)


class FaceEngineBase:
    """decode / detect_all / embed are the engine specific parts; `analyze_jpeg`
    is the per-frame result the API returns (identical for every engine, which
    is what tests monkeypatch)."""
    version = MODEL_VERSION

    def decode(self, data):
        raise NotImplementedError

    def image_size(self, img):
        return int(img.shape[1]), int(img.shape[0])

    def detect_all(self, img):
        raise NotImplementedError

    def embed(self, img, face):
        raise NotImplementedError

    def yaw(self, face):
        return landmark_yaw(face)

    def sharpness(self, img, face):
        """0..1 informational focus measure of the face area (None if unknown)."""
        return None

    def analyze_jpeg(self, data):
        """bytes -> dict with: ok, error?, faces (int), score, box, landmarks,
        yaw, area_ratio, quality, width, height, embedding.

        box / landmarks / yaw / area_ratio / quality / score describe the most
        confident face (None when there is none); `embedding` is returned only
        when EXACTLY ONE face was found (a crop with several people must never
        be turned into somebody's template). Coordinates are in the analysed
        image (already downscaled to <= 640 px), `width` / `height` are its size.
        `area_ratio` = face box area / (shorter image edge)^2."""
        out = {'ok': False, 'faces': 0, 'score': None, 'box': None, 'landmarks': None,
               'yaw': None, 'area_ratio': None, 'quality': None,
               'width': 0, 'height': 0, 'embedding': None}
        img = self.decode(data)
        if img is None:
            out['error'] = ERR_BAD_IMAGE
            return out
        width, height = self.image_size(img)
        out.update(ok=True, width=width, height=height)
        faces = self.detect_all(img)
        out['faces'] = len(faces)
        if not faces:
            return out
        best = max(faces, key=lambda f: f.score)
        short = float(min(width, height)) or 1.0
        x, y, w, h = best.box
        out['score'] = r(best.score, 4)
        out['box'] = [r(x, 2), r(y, 2), r(w, 2), r(h, 2)]
        out['landmarks'] = [[r(px, 2), r(py, 2)] for px, py in best.landmarks]
        out['yaw'] = r(self.yaw(best), 5)
        out['area_ratio'] = r(best.area / (short * short), 5)
        out['quality'] = self.sharpness(img, best)
        if len(faces) == 1:
            out['embedding'] = [r(v, 6) for v in self.embed(img, best)]
        return out


class OpenCVEngine(FaceEngineBase):
    """YuNet + SFace through cv2.FaceDetectorYN / cv2.FaceRecognizerSF.

    One instance is NOT thread safe (cv2 dnn objects keep state): the app keeps
    a pool of instances, one per worker thread."""

    def __init__(self, models_dir=None, score_threshold=0.8, nms_threshold=0.3, top_k=50,
                 verify_hashes=True, cv_threads=None):
        try:
            import cv2
            import numpy as np
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise EngineUnavailable('opencv-python-headless / numpy is not installed') from exc
        models_dir = models_dir or DEFAULT_MODELS_DIR
        detector_path = os.path.join(models_dir, DETECTOR_FILE)
        recognizer_path = os.path.join(models_dir, RECOGNIZER_FILE)
        for path in (detector_path, recognizer_path):
            if not os.path.isfile(path):
                raise EngineUnavailable('model file missing: %s' % path)
        self.model_hashes = {}
        if verify_hashes:
            for name, path in ((DETECTOR_FILE, detector_path), (RECOGNIZER_FILE, recognizer_path)):
                digest = file_sha256(path)
                if digest != EXPECTED_SHA256[name]:
                    raise EngineUnavailable('model file %s does not match the expected sha256' % name)
                self.model_hashes[name] = digest
        self._cv2 = cv2
        self._np = np
        if cv_threads:
            cv2.setNumThreads(int(cv_threads))
        try:
            self._detector = cv2.FaceDetectorYN.create(
                detector_path, '', (320, 320), score_threshold, nms_threshold, top_k)
            self._recognizer = cv2.FaceRecognizerSF.create(recognizer_path, '')
        except Exception as exc:  # cv2.error and friends
            raise EngineUnavailable('cannot load the OpenCV face models: %s' % exc) from exc
        self._lock = threading.Lock()      # belt and braces: one thread at a time per instance

    # -- images -----------------------------------------------------------
    def decode(self, data):
        if not data:
            return None
        dims = jpeg_dimensions(data)
        if dims is None or dims[0] <= 0 or dims[1] <= 0 or dims[0] * dims[1] > MAX_IMAGE_PIXELS:
            return None
        cv2, np = self._cv2, self._np
        try:
            img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            return None
        if img is None or img.size == 0:
            return None
        height, width = img.shape[:2]
        longest = max(height, width)
        if longest > MAX_ANALYSIS_EDGE:
            scale = MAX_ANALYSIS_EDGE / float(longest)
            img = cv2.resize(img, (max(int(width * scale), 1), max(int(height * scale), 1)),
                             interpolation=cv2.INTER_AREA)
        return img

    # -- detection / recognition -----------------------------------------
    def detect_all(self, img):
        height, width = img.shape[:2]
        with self._lock:
            self._detector.setInputSize((width, height))
            _, rows = self._detector.detect(img)
        if rows is None:
            return []
        faces = []
        for row in rows:
            faces.append(Face(
                box=(float(row[0]), float(row[1]), float(row[2]), float(row[3])),
                landmarks=[(float(row[4 + 2 * i]), float(row[5 + 2 * i])) for i in range(5)],
                score=float(row[14]),
                raw=row,
            ))
        return faces

    def embed(self, img, face):
        with self._lock:
            aligned = self._recognizer.alignCrop(img, face.raw)
            feature = self._recognizer.feature(aligned)
        return [float(v) for v in feature.flatten()]

    def warmup(self):
        """Run the detector once so the first real request is not slow."""
        self.detect_all(self._np.full((240, 320, 3), 127, dtype=self._np.uint8))

    def sharpness(self, img, face):
        """Variance of the Laplacian of the (96x96, grey) face crop mapped to
        0..1 (>= 1 at variance 400). Informational: the gateway does not use it
        for decisions yet."""
        cv2 = self._cv2
        try:
            height, width = img.shape[:2]
            x, y, w, h = face.box
            x0, y0 = int(max(x, 0)), int(max(y, 0))
            x1, y1 = int(min(x + w, width)), int(min(y + h, height))
            if x1 - x0 < 8 or y1 - y0 < 8:
                return 0.0
            crop = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
            crop = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_AREA)
            variance = float(cv2.Laplacian(crop, cv2.CV_64F).var())
            return r(min(variance / 400.0, 1.0), 4)
        except Exception:
            return None
