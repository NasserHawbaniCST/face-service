# -*- coding: utf-8 -*-
import json
import os
import sys
import time

import pytest
from fastapi.testclient import TestClient

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from app.config import load_settings  # noqa: E402
from app.main import create_app  # noqa: E402
from app.security import sign  # noqa: E402
from fake_engine import FakeEngine  # noqa: E402

KEY1 = 'k1-' + 'a' * 40
SECRET1 = 's1-' + 'b' * 40
KEY2 = 'k2-' + 'c' * 40
SECRET2 = 's2-' + 'd' * 40


def make_env(**extra):
    env = {'FACE_API_KEY_1': KEY1, 'FACE_HMAC_SECRET_1': SECRET1, 'FACE_WORKERS': '2',
           'FACE_LOG_LEVEL': 'INFO'}
    env.update({k: str(v) for k, v in extra.items()})
    return env


class Signer:
    """Builds the headers + exact body bytes of a valid request."""

    def __init__(self, key=KEY1, secret=SECRET1):
        self.key, self.secret = key, secret

    def request(self, payload, ts=None, key=None, secret=None, body=None, **headers):
        body = body if body is not None else json.dumps(payload).encode()
        ts_text = ('%.3f' % (time.time() if ts is None else ts))
        hdrs = {
            'Authorization': 'Bearer ' + (key or self.key),
            'X-Timestamp': ts_text,
            'X-Signature': sign(secret or self.secret, ts_text, body),
            'Content-Type': 'application/json',
        }
        hdrs.update(headers)
        return {'content': body, 'headers': hdrs}


@pytest.fixture
def signer():
    return Signer()


@pytest.fixture
def make_client():
    clients = []

    def factory(engine_factory=None, **env):
        settings = load_settings(make_env(**env))
        app = create_app(settings, engine_factory=engine_factory or (lambda: FakeEngine()))
        client = TestClient(app)
        client.__enter__()
        clients.append(client)
        return client

    yield factory
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def client(make_client):
    return make_client()
