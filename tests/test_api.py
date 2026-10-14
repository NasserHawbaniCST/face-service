# -*- coding: utf-8 -*-
"""Auth, limits and the analyze contract (deterministic fake engine)."""
import base64
import json
import logging
import time

import pytest

from app.engine import EngineUnavailable
from conftest import KEY1, KEY2, SECRET1, SECRET2, Signer
from fake_engine import fake_image

URL = '/v1/analyze'


def frame(fid='f1', **kw):
    return {'id': fid, 'image_b64': base64.b64encode(fake_image(**kw)).decode()}


def payload(*frames, request_id='req-1'):
    return {'request_id': request_id, 'frames': list(frames) or [frame()]}


def post(client, signer, data=None, **kw):
    return client.post(URL, **signer.request(data if data is not None else payload(), **kw))


def code(resp):
    return resp.json()['error']['code']


# ---------------------------------------------------------------- health
def test_health_needs_no_auth_and_reports_versions(client):
    resp = client.get('/health')
    assert resp.status_code == 200
    body = resp.json()
    assert body['status'] == 'ok'
    assert body['engine_version'] == 'yunet_2023mar+sface_2021dec'
    assert body['service_version']
    assert set(body['models']) == {'detector', 'recognizer'}
    assert 'X-Request-Id' in resp.headers


def test_health_reports_engine_down(make_client):
    def broken():
        raise EngineUnavailable('model file missing: x')
    client = make_client(engine_factory=broken)
    resp = client.get('/health')
    assert resp.status_code == 503
    assert resp.json()['status'] == 'unavailable'
    assert 'model file missing' in resp.json()['error']


def test_analyze_when_engine_down_is_503(make_client, signer):
    def broken():
        raise EngineUnavailable('nope')
    client = make_client(engine_factory=broken)
    resp = post(client, signer)
    assert (resp.status_code, code(resp)) == (503, 'ENGINE_UNAVAILABLE')


# ---------------------------------------------------------------- auth
def test_valid_request(client, signer):
    resp = post(client, signer)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body['request_id'] == 'req-1'
    assert body['engine_version'] == 'yunet_2023mar+sface_2021dec'
    assert len(body['frames']) == 1


def test_missing_and_wrong_key(client, signer):
    resp = client.post(URL, content=b'{}', headers={'Content-Type': 'application/json'})
    assert (resp.status_code, code(resp)) == (401, 'UNAUTHORIZED')
    resp = post(client, signer, key='x' * 40)
    assert (resp.status_code, code(resp)) == (401, 'UNAUTHORIZED')
    resp = client.post(URL, **dict(signer.request(payload()), headers={
        **signer.request(payload())['headers'], 'Authorization': 'Basic abc'}))
    assert code(resp) == 'UNAUTHORIZED'


def test_bad_signature_wrong_secret_and_tampered_body(client, signer):
    resp = post(client, signer, secret='z' * 40)
    assert (resp.status_code, code(resp)) == (401, 'BAD_SIGNATURE')
    ok = signer.request(payload())
    tampered = ok['content'].replace(b'req-1', b'req-2')
    resp = client.post(URL, content=tampered, headers=ok['headers'])
    assert code(resp) == 'BAD_SIGNATURE'
    headers = dict(ok['headers'])
    del headers['X-Signature']
    assert code(client.post(URL, content=ok['content'], headers=headers)) == 'BAD_SIGNATURE'


def test_stale_and_future_and_garbage_timestamp(client, signer):
    assert code(post(client, signer, ts=time.time() - 120)) == 'STALE_TIMESTAMP'
    assert code(post(client, signer, ts=time.time() + 120)) == 'STALE_TIMESTAMP'
    # inside the tolerance is fine
    assert post(client, signer, ts=time.time() - 30).status_code == 200
    req = signer.request(payload())
    req['headers']['X-Timestamp'] = 'yesterday'
    assert code(client.post(URL, **req)) == 'STALE_TIMESTAMP'
    req['headers'].pop('X-Timestamp')
    assert code(client.post(URL, **req)) == 'STALE_TIMESTAMP'


def test_replay_is_refused(client, signer):
    req = signer.request(payload())
    assert client.post(URL, **req).status_code == 200
    again = client.post(URL, **req)
    assert (again.status_code, code(again)) == (401, 'REPLAYED_REQUEST')


def test_rotation_second_key_works_and_each_key_has_its_own_secret(make_client):
    client = make_client(FACE_API_KEY_2=KEY2, FACE_HMAC_SECRET_2=SECRET2)
    assert post(client, Signer(KEY1, SECRET1)).status_code == 200
    assert post(client, Signer(KEY2, SECRET2)).status_code == 200
    # key 2 with key 1's secret is rejected
    assert code(post(client, Signer(KEY2, SECRET1))) == 'BAD_SIGNATURE'


def test_weak_or_missing_credentials_refuse_to_start():
    from app.config import ConfigError, load_settings
    with pytest.raises(ConfigError):
        load_settings({})
    with pytest.raises(ConfigError):
        load_settings({'FACE_API_KEY_1': 'short', 'FACE_HMAC_SECRET_1': SECRET1})
    with pytest.raises(ConfigError):
        load_settings({'FACE_API_KEY_1': KEY1})
    with pytest.raises(ConfigError):
        load_settings({'FACE_API_KEY_1': KEY1, 'FACE_HMAC_SECRET_1': SECRET1, 'FACE_ALLOWED_IPS': 'nonsense'})


def test_ip_allowlist(make_client, signer):
    client = make_client(FACE_ALLOWED_IPS='10.1.2.3, 192.168.0.0/16')
    resp = post(client, signer)
    assert (resp.status_code, code(resp)) == (403, 'FORBIDDEN_IP')   # testclient peer is 'testclient'
    assert client.get('/health').status_code == 200                 # health stays open


def test_forwarded_for_only_trusted_from_trusted_proxies():
    from app.config import load_settings
    from app.security import Authenticator
    from conftest import make_env
    plain = Authenticator(load_settings(make_env()))
    assert plain.client_ip('1.1.1.1', '9.9.9.9') == '1.1.1.1'
    proxied = Authenticator(load_settings(make_env(FACE_TRUSTED_PROXIES='127.0.0.1')))
    assert proxied.client_ip('127.0.0.1', '8.8.8.8, 127.0.0.1') == '8.8.8.8'
    assert proxied.client_ip('5.5.5.5', '8.8.8.8') == '5.5.5.5'       # untrusted peer: header ignored
    # a spoofed leading entry does not help (right-most untrusted hop wins)
    assert proxied.client_ip('127.0.0.1', '1.2.3.4, 8.8.8.8') == '8.8.8.8'


def test_rate_limit_per_key(make_client, signer):
    client = make_client(FACE_RATE_LIMIT_PER_MIN=3)
    for _ in range(3):
        assert post(client, signer).status_code == 200
    resp = post(client, signer)
    assert (resp.status_code, code(resp)) == (429, 'RATE_LIMITED')
    assert int(resp.headers['Retry-After']) >= 1


def test_failed_auth_attempts_are_throttled_per_ip(make_client, signer):
    client = make_client(FACE_AUTH_FAIL_LIMIT_PER_MIN=3)
    for _ in range(3):
        assert code(post(client, signer, key='x' * 40)) == 'UNAUTHORIZED'
    resp = post(client, signer)       # even a valid request is throttled from that address
    assert (resp.status_code, code(resp)) == (429, 'RATE_LIMITED')


# ---------------------------------------------------------------- limits
def test_oversized_body_by_content_length(make_client, signer):
    client = make_client(FACE_MAX_BODY_BYTES=2048)
    big = payload(frame(faces=[]), request_id='x')
    big['pad'] = 'p' * 5000
    resp = post(client, signer, big)
    assert (resp.status_code, code(resp)) == (413, 'PAYLOAD_TOO_LARGE')
    assert 'X-Request-Id' in resp.headers


def test_oversized_streamed_body_without_content_length(make_client, signer):
    client = make_client(FACE_MAX_BODY_BYTES=2048)
    req = signer.request(payload())
    chunks = (b'x' * 1000 for _ in range(10))        # generator => chunked, no Content-Length
    resp = client.post(URL, content=chunks, headers={k: v for k, v in req['headers'].items()})
    assert (resp.status_code, code(resp)) == (413, 'PAYLOAD_TOO_LARGE')


def test_too_many_frames(client, signer):
    seven = [frame('f%d' % i) for i in range(7)]
    resp = post(client, signer, payload(*seven))
    assert (resp.status_code, code(resp)) == (400, 'TOO_MANY_FRAMES')
    six = [frame('f%d' % i) for i in range(6)]
    assert post(client, signer, payload(*six)).status_code == 200


def test_single_frame_too_large(make_client, signer):
    client = make_client(FACE_MAX_FRAME_BYTES=2048)
    big = {'id': 'big', 'image_b64': base64.b64encode(b'a' * 3000).decode()}
    resp = post(client, signer, payload(big))
    assert (resp.status_code, code(resp)) == (413, 'FRAME_TOO_LARGE')


@pytest.mark.parametrize('body', [
    b'not json', b'[]', b'{}',
    json.dumps({'request_id': 'r', 'frames': []}).encode(),
    json.dumps({'request_id': 'bad id!', 'frames': [{'id': 'a', 'image_b64': 'AA=='}]}).encode(),
    json.dumps({'request_id': 'r', 'frames': [{'id': 'a'}]}).encode(),
    json.dumps({'request_id': 'r', 'frames': [{'id': 'a', 'image_b64': 'AA=='},
                                              {'id': 'a', 'image_b64': 'AA=='}]}).encode(),
])
def test_malformed_requests(client, signer, body):
    resp = post(client, signer, body=body)
    assert (resp.status_code, code(resp)) == (400, 'BAD_REQUEST')


def test_request_id_echo_and_generation(client, signer):
    resp = client.post(URL, **signer.request(payload(), **{'X-Request-Id': 'trace-123'}))
    assert resp.headers['X-Request-Id'] == 'trace-123'
    resp = client.post(URL, **signer.request(payload(), **{'X-Request-Id': 'bad id with spaces'}))
    assert resp.headers['X-Request-Id'] not in ('', 'bad id with spaces')
    # errors carry it too, in header and body
    err = client.post(URL, content=b'{}', headers={'X-Request-Id': 'abc-1'})
    assert err.headers['X-Request-Id'] == 'abc-1' and err.json()['request_id'] == 'abc-1'


def test_unknown_route_and_method(client):
    assert client.get('/nope').status_code == 404
    assert client.get(URL).status_code == 405


# ---------------------------------------------------------------- analyze
def test_deterministic_faces_shape_and_semantics(client, signer):
    resp = post(client, signer, payload(
        frame('a', person='alice', yaw=0.0, area=10000),
        frame('b', person='alice', yaw=-0.35, area=10000, seed=1),
        frame('c', faces=[]),
        frame('d', faces=[{'p': 'alice', 'yaw': 0, 'area': 10000}, {'p': 'bob', 'yaw': 0, 'area': 9000, 'score': 0.9}]),
    ))
    assert resp.status_code == 200
    frames = {f['id']: f for f in resp.json()['frames']}
    assert list(frames) == ['a', 'b', 'c', 'd']          # order preserved
    a = frames['a']
    assert a['ok'] is True and a['faces'] == 1
    assert a['box'] == [100.0, 100.0, 100.0, 100.0]
    assert len(a['landmarks']) == 5
    assert a['yaw'] == pytest.approx(0.0, abs=1e-4)
    assert a['width'] == 480 and a['height'] == 640
    assert a['area_ratio'] == pytest.approx(10000 / (480 * 480), abs=1e-4)   # short edge = 480
    assert len(a['embedding']) == 128
    assert frames['b']['yaw'] == pytest.approx(-0.35, abs=1e-4)
    # no face
    assert frames['c']['ok'] is True and frames['c']['faces'] == 0
    assert frames['c']['embedding'] is None and frames['c']['box'] is None
    # several faces: counted, described by the most confident one, but NO embedding
    assert frames['d']['faces'] == 2 and frames['d']['embedding'] is None
    assert frames['d']['score'] == pytest.approx(0.99)


def test_unusable_frames_are_reported_per_frame(client, signer):
    junk = {'id': 'junk', 'image_b64': base64.b64encode(b'not an image').decode()}
    bad64 = {'id': 'b64', 'image_b64': '***not base64***'}
    resp = post(client, signer, payload(frame('good'), junk, bad64))
    assert resp.status_code == 200
    got = {f['id']: f for f in resp.json()['frames']}
    assert got['good']['ok'] is True
    assert (got['junk']['ok'], got['junk']['error']) == (False, 'bad_image')
    assert (got['b64']['ok'], got['b64']['error']) == (False, 'bad_base64')
    assert got['junk']['embedding'] is None


def test_engine_exception_is_contained(make_client, signer):
    from fake_engine import FakeEngine

    class Exploding(FakeEngine):
        def embed(self, img, face):
            raise RuntimeError('boom with secret pixels')

    client = make_client(engine_factory=lambda: Exploding())
    resp = post(client, signer)
    assert resp.status_code == 200
    f = resp.json()['frames'][0]
    assert (f['ok'], f['error']) == (False, 'engine_error')
    assert 'boom' not in resp.text


def test_busy_when_queue_is_full(make_client, signer):
    from fake_engine import FakeEngine
    client = make_client(engine_factory=lambda: FakeEngine(), FACE_WORKERS=1, FACE_MAX_QUEUE=0)
    client.app.state.pool.pending = 1                      # one request "in flight"
    resp = post(client, signer)
    assert (resp.status_code, code(resp)) == (503, 'BUSY')
    assert resp.headers['Retry-After']


def test_nothing_sensitive_is_logged(client, signer, caplog):
    caplog.set_level(logging.DEBUG)
    img = frame('f', person='alice')
    resp = post(client, signer, payload(img, request_id='log-check'))
    emb = resp.json()['frames'][0]['embedding']
    post(client, signer, body=b'{"image_b64": "SECRETBODY"}')       # rejected request
    text = '\n'.join(r.getMessage() for r in caplog.records)
    assert 'log-check' in text and 'status=200' in text
    assert img['image_b64'][:40] not in text
    assert str(emb[0]) not in text and 'SECRETBODY' not in text
    assert 'embedding' not in text and SECRET1 not in text and KEY1 not in text


# ---------------------------------------------------------------- the event loop stays free
def test_event_loop_stays_responsive_while_analysing(make_client):
    import asyncio
    import httpx
    from fake_engine import FakeEngine
    from conftest import make_env
    from app.config import load_settings
    from app.main import create_app

    app = create_app(load_settings(make_env(FACE_WORKERS=2)), engine_factory=lambda: FakeEngine(delay=0.4))
    signer = Signer()

    async def scenario():
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url='http://t') as ac:
                slow = [asyncio.create_task(ac.post(URL, **signer.request(payload(request_id='r%d' % i))))
                        for i in range(2)]
                await asyncio.sleep(0.1)
                t0 = time.monotonic()
                health = await ac.get('/health')
                health_ms = (time.monotonic() - t0) * 1000
                done = await asyncio.gather(*slow)
                return health, health_ms, done

    health, health_ms, done = asyncio.run(scenario())
    assert health.status_code == 200
    assert health_ms < 300                    # not blocked behind the 400 ms analyses
    assert [r.status_code for r in done] == [200, 200]
