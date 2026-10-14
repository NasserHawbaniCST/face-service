# -*- coding: utf-8 -*-
"""Deterministic engine for tests and for the gateway's end-to-end test.

Its "images" are tiny JSON documents (the same format the gateway addon's
FakeEngine uses) describing the faces in them:
    {"w": 480, "h": 640, "faces": [{"p": "alice", "yaw": 0.0, "area": 10000, "seed": 0, "score": 0.99}]}
The service computes yaw from the landmarks this engine fabricates, so the
numbers that come back through the API match what the document asked for.
"""
import hashlib
import json
import random

from app.engine import Face, FaceEngineBase


def person_vector(name, dim=128):
    seed = int(hashlib.sha256(name.encode()).hexdigest()[:12], 16)
    rnd = random.Random(seed)
    return [rnd.uniform(-1, 1) for _ in range(dim)]


def jitter(vec, noise, seed):
    rnd = random.Random(seed)
    return [v + rnd.uniform(-noise, noise) for v in vec]


def fake_image(person='alice', yaw=0.0, area=10000, seed=0, score=0.99, faces=None, width=480, height=640):
    if faces is None:
        faces = [{'p': person, 'yaw': yaw, 'area': area, 'seed': seed, 'score': score}]
    return json.dumps({'w': width, 'h': height, 'faces': faces}).encode()


class FakeEngine(FaceEngineBase):
    version = 'yunet_2023mar+sface_2021dec'
    model_hashes = {'face_detection_yunet_2023mar.onnx': 'fake', 'face_recognition_sface_2021dec.onnx': 'fake'}

    def __init__(self, noise=0.03, delay=0.0):
        self.noise = noise
        self.delay = delay
        self.calls = 0

    def decode(self, data):
        self.calls += 1
        if self.delay:
            import time
            time.sleep(self.delay)
        if data[:2] == b'\xff\xd8':        # a real JPEG (e.g. the gateway's "Test connection" picture)
            return {'w': 64, 'h': 64, 'faces': []}
        try:
            doc = json.loads(data.decode())
        except Exception:
            return None
        return doc if isinstance(doc, dict) and 'faces' in doc else None

    def image_size(self, img):
        return img['w'], img['h']

    def detect_all(self, img):
        faces = []
        for raw in img['faces']:
            side = raw['area'] ** 0.5
            # right eye / left eye / nose such that the yaw ratio equals raw['yaw']
            landmarks = [(30.0, 40.0), (70.0, 40.0), (50.0 + raw.get('yaw', 0.0) * 40.0, 60.0),
                         (35.0, 80.0), (65.0, 80.0)]
            faces.append(Face(box=(100.0, 100.0, side, side), landmarks=landmarks,
                              score=raw.get('score', 0.99), raw=raw))
        return faces

    def embed(self, img, face):
        return jitter(person_vector(face.raw['p']), self.noise, face.raw.get('seed', 0))
