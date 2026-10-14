# -*- coding: utf-8 -*-
"""Settings, read once from environment variables (prefix FACE_).

Nothing is read from disk except the two model files and (optionally) a
`.env` file that the process manager / docker loads for us - the app itself
only looks at `os.environ`.
"""
import ipaddress
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

MIN_SECRET_LEN = 32


class ConfigError(Exception):
    """Invalid configuration - the service refuses to start."""


def _int(env, name, default, minimum=None, maximum=None):
    raw = env.get(name)
    if raw is None or raw.strip() == '':
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        raise ConfigError('%s must be an integer, got %r' % (name, raw))
    if minimum is not None and value < minimum:
        raise ConfigError('%s must be >= %s' % (name, minimum))
    if maximum is not None and value > maximum:
        raise ConfigError('%s must be <= %s' % (name, maximum))
    return value


def _float(env, name, default):
    raw = env.get(name)
    if raw is None or raw.strip() == '':
        return default
    try:
        return float(raw.strip())
    except ValueError:
        raise ConfigError('%s must be a number, got %r' % (name, raw))


def _networks(raw, name):
    nets = []
    for part in (raw or '').replace(';', ',').split(','):
        part = part.strip()
        if not part:
            continue
        try:
            nets.append(ipaddress.ip_network(part, strict=False))
        except ValueError:
            raise ConfigError('%s: %r is not an IP address / CIDR' % (name, part))
    return nets


@dataclass
class ApiKey:
    """One accepted credential pair (two slots exist so keys can be rotated
    without downtime)."""
    slot: str
    key: str
    secret: str


@dataclass
class Settings:
    keys: List[ApiKey] = field(default_factory=list)
    allowed_networks: list = field(default_factory=list)
    trusted_proxies: list = field(default_factory=list)
    max_body_bytes: int = 4 * 1024 * 1024
    max_frames: int = 6
    max_frame_bytes: int = 400 * 1024          # decoded
    ts_tolerance_s: float = 60.0
    rate_limit_per_min: int = 120              # per API key
    auth_fail_limit_per_min: int = 30          # per client IP, failed authentications
    workers: int = 2                           # engines + threads
    cv_threads: int = 1                        # OpenCV intra-op threads per engine
    max_queue: int = 4                         # requests waiting for a free engine
    request_timeout_s: float = 20.0
    models_dir: Optional[str] = None
    log_level: str = 'INFO'

    @property
    def replay_ttl_s(self):
        return self.ts_tolerance_s * 2 + 5


def default_workers():
    cpus = os.cpu_count() or 1
    return max(1, min(cpus, 4))


def load_settings(env=None) -> Settings:
    """Build Settings from `env` (default os.environ). Raises ConfigError."""
    env = os.environ if env is None else env
    s = Settings()

    shared_secret = (env.get('FACE_HMAC_SECRET') or '').strip()
    for slot in ('1', '2'):
        key = (env.get('FACE_API_KEY_%s' % slot) or '').strip()
        if not key:
            continue
        secret = (env.get('FACE_HMAC_SECRET_%s' % slot) or '').strip() or shared_secret
        if len(key) < MIN_SECRET_LEN:
            raise ConfigError('FACE_API_KEY_%s must be at least %d characters '
                              '(generate one with: openssl rand -hex 32)' % (slot, MIN_SECRET_LEN))
        if len(secret) < MIN_SECRET_LEN:
            raise ConfigError('FACE_HMAC_SECRET_%s (or FACE_HMAC_SECRET) must be at least %d characters'
                              % (slot, MIN_SECRET_LEN))
        s.keys.append(ApiKey(slot=slot, key=key, secret=secret))
    if not s.keys:
        raise ConfigError('no credentials: set FACE_API_KEY_1 and FACE_HMAC_SECRET_1')
    if len(s.keys) == 2 and s.keys[0].key == s.keys[1].key:
        raise ConfigError('FACE_API_KEY_1 and FACE_API_KEY_2 must differ')

    s.allowed_networks = _networks(env.get('FACE_ALLOWED_IPS'), 'FACE_ALLOWED_IPS')
    s.trusted_proxies = _networks(env.get('FACE_TRUSTED_PROXIES'), 'FACE_TRUSTED_PROXIES')
    s.max_body_bytes = _int(env, 'FACE_MAX_BODY_BYTES', s.max_body_bytes, 1024, 64 * 1024 * 1024)
    s.max_frames = _int(env, 'FACE_MAX_FRAMES', s.max_frames, 1, 6)
    s.max_frame_bytes = _int(env, 'FACE_MAX_FRAME_BYTES', s.max_frame_bytes, 1024, 4 * 1024 * 1024)
    s.ts_tolerance_s = _float(env, 'FACE_TS_TOLERANCE', s.ts_tolerance_s)
    s.rate_limit_per_min = _int(env, 'FACE_RATE_LIMIT_PER_MIN', s.rate_limit_per_min, 1)
    s.auth_fail_limit_per_min = _int(env, 'FACE_AUTH_FAIL_LIMIT_PER_MIN', s.auth_fail_limit_per_min, 1)
    s.workers = _int(env, 'FACE_WORKERS', default_workers(), 1, 32)
    s.cv_threads = _int(env, 'FACE_CV_THREADS', s.cv_threads, 1, 32)
    s.max_queue = _int(env, 'FACE_MAX_QUEUE', max(s.workers * 2, 2), 0, 1000)
    s.request_timeout_s = _float(env, 'FACE_REQUEST_TIMEOUT', s.request_timeout_s)
    s.models_dir = (env.get('FACE_MODELS_DIR') or '').strip() or None
    s.log_level = (env.get('FACE_LOG_LEVEL') or 'INFO').strip().upper()
    return s


def ip_in(addr: str, networks) -> bool:
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return False
    return any(ip in net for net in networks)
