# -*- coding: utf-8 -*-
"""FastAPI application: GET /health and POST /v1/analyze.

Privacy rules enforced here: images, embeddings and request bodies are never
logged and never stored (one log line per request: id, status, timing, counts).
The service is stateless - it holds no customer credentials and no sessions.

Run:  uvicorn --factory app.main:create_app --host 127.0.0.1 --port 8765 --no-access-log
(ONE process: the replay cache and rate limits live in memory; scale with
FACE_WORKERS, see README.)
"""
import asyncio
import base64
import binascii
import json
import logging
import queue
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import SERVICE_VERSION
from .config import Settings, load_settings
from .engine import (
    DETECTOR_FILE, ERR_BAD_IMAGE, MODEL_VERSION, RECOGNIZER_FILE, EngineUnavailable, OpenCVEngine,
)
from .security import ApiError, Authenticator

log = logging.getLogger('face_service')

ID_RE = re.compile(r'^[A-Za-z0-9._:\-]{1,64}$')
ERR_BAD_BASE64 = 'bad_base64'
ERR_ENGINE = 'engine_error'


class BodyTooLarge(Exception):
    """Raised by the ASGI receive wrapper when the body exceeds the limit."""


# ----------------------------------------------------------------------
# ASGI layer: request id, strict body limit, one log line per request
# ----------------------------------------------------------------------
class GuardMiddleware:

    def __init__(self, app, max_body_bytes):
        self.app = app
        self.max_body = max_body_bytes

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            await self.app(scope, receive, send)
            return
        headers = {k: v for k, v in scope.get('headers', [])}
        rid = headers.get(b'x-request-id', b'').decode('latin-1').strip()
        if not ID_RE.match(rid):
            rid = uuid.uuid4().hex
        state = scope.setdefault('state', {})
        state['request_id'] = rid
        state['log'] = {}
        started = time.monotonic()
        status_holder = {'status': 0}

        async def send_wrapper(message):
            if message['type'] == 'http.response.start':
                status_holder['status'] = message['status']
                hdrs = [(k, v) for k, v in message.get('headers', []) if k.lower() != b'x-request-id']
                hdrs.append((b'x-request-id', rid.encode('ascii')))
                message = dict(message, headers=hdrs)
            await send(message)

        content_length = headers.get(b'content-length', b'')
        if content_length.isdigit() and int(content_length) > self.max_body:
            # refuse without reading a single byte of the body
            body = json.dumps({'error': {'code': 'PAYLOAD_TOO_LARGE', 'message': 'request body too large'},
                               'request_id': rid}).encode()
            await send_wrapper({'type': 'http.response.start', 'status': 413, 'headers': [
                (b'content-type', b'application/json'), (b'content-length', str(len(body)).encode())]})
            await send_wrapper({'type': 'http.response.body', 'body': body})
            state['log']['code'] = 'PAYLOAD_TOO_LARGE'
        else:
            total = 0

            async def limited_receive():
                nonlocal total
                message = await receive()
                if message['type'] == 'http.request':
                    total += len(message.get('body', b''))
                    if total > self.max_body:
                        raise BodyTooLarge()
                return message

            try:
                await self.app(scope, limited_receive, send_wrapper)
            finally:
                self._log(scope, rid, status_holder['status'], started)
            return
        self._log(scope, rid, status_holder['status'], started)

    @staticmethod
    def _log(scope, rid, status, started):
        extra = scope.get('state', {}).get('log', {})
        client = scope.get('client')
        parts = ['rid=%s' % rid, '%s %s' % (scope.get('method', '-'), scope.get('path', '-')),
                 'status=%s' % status, 'ms=%d' % int((time.monotonic() - started) * 1000),
                 'ip=%s' % (client[0] if client else '-')]
        parts += ['%s=%s' % (k, v) for k, v in extra.items()]
        log.info(' '.join(parts))


# ----------------------------------------------------------------------
# engine pool
# ----------------------------------------------------------------------
class EnginePool:
    """`workers` engine instances + `workers` threads: an engine is never used
    by two threads at once, and the event loop is never blocked by OpenCV."""

    def __init__(self, settings: Settings, factory):
        self.settings = settings
        self.factory = factory
        self.engines = queue.Queue()
        self.executor = None
        self.error = None
        self.pending = 0
        self.model_hashes = {}
        self.version = MODEL_VERSION

    def start(self):
        try:
            engines = [self.factory() for _ in range(self.settings.workers)]
        except EngineUnavailable as exc:
            self.error = str(exc)
            log.error('face engine unavailable: %s', exc)
            return
        except Exception as exc:  # noqa: BLE001
            self.error = 'engine failed to start: %s' % type(exc).__name__
            log.exception('face engine failed to start')
            return
        for engine in engines:
            warm = getattr(engine, 'warmup', None)
            if warm:
                try:
                    warm()
                except Exception:  # noqa: BLE001
                    log.warning('engine warm-up failed', exc_info=True)
            self.engines.put(engine)
        first = engines[0]
        self.model_hashes = dict(getattr(first, 'model_hashes', {}) or {})
        self.version = getattr(first, 'version', MODEL_VERSION)
        self.executor = ThreadPoolExecutor(max_workers=len(engines), thread_name_prefix='face')
        log.info('face engine ready: version=%s workers=%d', self.version, len(engines))

    def stop(self):
        if self.executor:
            self.executor.shutdown(wait=False, cancel_futures=True)

    @property
    def available(self):
        return self.executor is not None

    def process(self, datas):
        """Runs in a worker thread: one engine for the whole request."""
        engine = self.engines.get()
        try:
            results = []
            for data in datas:
                if data is None:
                    results.append(None)
                    continue
                try:
                    results.append(engine.analyze_jpeg(data))
                except Exception as exc:  # noqa: BLE001 - never leak details / pixels
                    log.warning('frame analysis failed: %s', type(exc).__name__)
                    results.append({'ok': False, 'error': ERR_ENGINE, 'faces': 0, 'score': None,
                                    'box': None, 'landmarks': None, 'yaw': None, 'area_ratio': None,
                                    'quality': None, 'width': 0, 'height': 0, 'embedding': None})
            return results
        finally:
            self.engines.put(engine)


def opencv_factory(settings):
    def make():
        return OpenCVEngine(models_dir=settings.models_dir, cv_threads=settings.cv_threads)
    return make


# ----------------------------------------------------------------------
# request validation
# ----------------------------------------------------------------------
def parse_request(raw, settings):
    """raw body bytes -> (request_id, [(frame_id, bytes|None, error|None)])."""
    try:
        doc = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        raise ApiError(400, 'BAD_REQUEST', 'body is not valid JSON')
    if not isinstance(doc, dict):
        raise ApiError(400, 'BAD_REQUEST', 'body must be a JSON object')
    request_id = doc.get('request_id')
    if not isinstance(request_id, str) or not ID_RE.match(request_id):
        raise ApiError(400, 'BAD_REQUEST', 'request_id must be 1-64 characters of [A-Za-z0-9._:-]')
    frames = doc.get('frames')
    if not isinstance(frames, list) or not frames:
        raise ApiError(400, 'BAD_REQUEST', 'frames must be a non-empty list')
    if len(frames) > settings.max_frames:
        raise ApiError(400, 'TOO_MANY_FRAMES', 'at most %d frames per request' % settings.max_frames)
    max_b64 = ((settings.max_frame_bytes + 2) // 3) * 4 + 4
    parsed, seen = [], set()
    for frame in frames:
        if not isinstance(frame, dict):
            raise ApiError(400, 'BAD_REQUEST', 'each frame must be an object')
        fid, encoded = frame.get('id'), frame.get('image_b64')
        if not isinstance(fid, str) or not ID_RE.match(fid) or fid in seen:
            raise ApiError(400, 'BAD_REQUEST', 'frame id must be unique, 1-64 characters of [A-Za-z0-9._:-]')
        seen.add(fid)
        if not isinstance(encoded, str) or not encoded:
            raise ApiError(400, 'BAD_REQUEST', 'image_b64 must be a non-empty string')
        if len(encoded) > max_b64:
            raise ApiError(413, 'FRAME_TOO_LARGE', 'a frame exceeds %d bytes' % settings.max_frame_bytes)
        try:
            data = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError):
            parsed.append((fid, None, ERR_BAD_BASE64))
            continue
        if len(data) > settings.max_frame_bytes:
            raise ApiError(413, 'FRAME_TOO_LARGE', 'a frame exceeds %d bytes' % settings.max_frame_bytes)
        parsed.append((fid, data, None))
    return request_id, parsed


# ----------------------------------------------------------------------
def error_response(request: Request, status, code, message, retry_after=None):
    headers = {'Retry-After': str(retry_after)} if retry_after else None
    rid = request.scope.get('state', {}).get('request_id')
    request.scope.setdefault('state', {}).setdefault('log', {})['code'] = code
    return JSONResponse({'error': {'code': code, 'message': message}, 'request_id': rid},
                        status_code=status, headers=headers)


def create_app(settings: Settings = None, engine_factory=None) -> FastAPI:
    """`engine_factory()` -> a FaceEngineBase (tests pass a deterministic fake)."""
    settings = settings or load_settings()
    logging.basicConfig(level=getattr(logging, settings.log_level, logging.INFO),
                        format='%(asctime)s %(levelname)s %(name)s %(message)s')
    auth = Authenticator(settings)
    pool = EnginePool(settings, engine_factory or opencv_factory(settings))

    @asynccontextmanager
    async def lifespan(app):
        await asyncio.get_running_loop().run_in_executor(None, pool.start)
        try:
            yield
        finally:
            pool.stop()

    app = FastAPI(title='Face service', version=SERVICE_VERSION, lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.state.settings, app.state.auth, app.state.pool = settings, auth, pool
    app.add_middleware(GuardMiddleware, max_body_bytes=settings.max_body_bytes)

    # ---- error handlers ------------------------------------------------
    @app.exception_handler(ApiError)
    async def _api_error(request, exc):
        return error_response(request, exc.status, exc.code, exc.message, exc.retry_after)

    @app.exception_handler(BodyTooLarge)
    async def _too_large(request, exc):
        return error_response(request, 413, 'PAYLOAD_TOO_LARGE', 'request body too large')

    @app.exception_handler(StarletteHTTPException)
    async def _http(request, exc):
        codes = {404: 'NOT_FOUND', 405: 'METHOD_NOT_ALLOWED'}
        return error_response(request, exc.status_code, codes.get(exc.status_code, 'HTTP_ERROR'),
                              'not found' if exc.status_code == 404 else 'request not allowed')

    @app.exception_handler(Exception)
    async def _unexpected(request, exc):
        log.error('unhandled error: %s', type(exc).__name__)
        return error_response(request, 500, 'INTERNAL_ERROR', 'internal error')

    # ---- routes ----------------------------------------------------------
    @app.get('/health')
    async def health(request: Request):
        body = {
            'status': 'ok' if pool.available else 'unavailable',
            'service_version': SERVICE_VERSION,
            'engine_version': pool.version,
            'models': {
                'detector': {'file': DETECTOR_FILE, 'sha256': pool.model_hashes.get(DETECTOR_FILE)},
                'recognizer': {'file': RECOGNIZER_FILE, 'sha256': pool.model_hashes.get(RECOGNIZER_FILE)},
            },
            'workers': settings.workers,
        }
        if not pool.available:
            body['error'] = pool.error or 'engine not started'
        return JSONResponse(body, status_code=200 if pool.available else 503)

    @app.post('/v1/analyze')
    async def analyze(request: Request):
        log_extra = request.scope['state']['log']
        headers = request.headers
        ip = auth.client_ip(request.client.host if request.client else '', headers.get('x-forwarded-for'))
        auth.check_ip(ip)
        raw = await request.body()
        key = auth.authenticate(ip, headers.get('authorization'), headers.get('x-timestamp'),
                                headers.get('x-signature'), raw)
        log_extra['key'] = key.slot
        request_id, frames = parse_request(raw, settings)
        log_extra['req'] = request_id
        log_extra['frames'] = len(frames)
        if not pool.available:
            raise ApiError(503, 'ENGINE_UNAVAILABLE', 'face engine is not available', retry_after=5)
        if pool.pending >= settings.workers + settings.max_queue:
            raise ApiError(503, 'BUSY', 'service is busy, retry shortly', retry_after=1)
        pool.pending += 1
        try:
            loop = asyncio.get_running_loop()
            results = await asyncio.wait_for(
                loop.run_in_executor(pool.executor, pool.process, [d for _, d, _ in frames]),
                timeout=settings.request_timeout_s)
        except asyncio.TimeoutError:
            raise ApiError(504, 'TIMEOUT', 'analysis took too long')
        finally:
            pool.pending -= 1
        out = []
        for (fid, _data, err), res in zip(frames, results):
            if res is None:
                res = {'ok': False, 'error': err or ERR_BAD_IMAGE, 'faces': 0, 'score': None, 'box': None,
                       'landmarks': None, 'yaw': None, 'area_ratio': None, 'quality': None,
                       'width': 0, 'height': 0, 'embedding': None}
            entry = {'id': fid}
            entry.update(res)
            out.append(entry)
        log_extra['faces'] = ','.join(str(f['faces']) for f in out)
        log_extra['ok'] = sum(1 for f in out if f['ok'])
        return JSONResponse({'request_id': request_id, 'engine_version': pool.version, 'frames': out})

    return app
