# -*- coding: utf-8 -*-
"""Authentication + abuse controls (framework independent, unit testable).

Every /v1 request must carry
    Authorization: Bearer <api key>
    X-Timestamp:   <unix seconds, float ok>
    X-Signature:   sha256=<hex HMAC-SHA256(secret, timestamp + '.' + raw_body)>
so a captured request cannot be replayed (timestamp window + one-time
signature cache) nor altered (the body is signed).
"""
import hashlib
import hmac
import threading
import time
from collections import deque, OrderedDict

from .config import ip_in


class ApiError(Exception):
    """Carries a stable machine readable `code`, the HTTP status and a short,
    generic message (never anything derived from the request body)."""

    def __init__(self, status, code, message, retry_after=None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.retry_after = retry_after


def sign(secret, timestamp, body):
    """The signature header value for (timestamp string, raw body bytes)."""
    if isinstance(timestamp, (int, float)):
        timestamp = repr(timestamp) if isinstance(timestamp, float) else str(timestamp)
    mac = hmac.new(secret.encode('utf-8'), timestamp.encode('utf-8') + b'.' + body, hashlib.sha256)
    return 'sha256=' + mac.hexdigest()


def _digest(text):
    return hashlib.sha256(text.encode('utf-8')).digest()


class SlidingWindow:
    """Thread safe "at most N events per `window` seconds" counter per key."""

    def __init__(self, limit, window=60.0, clock=time.monotonic, max_keys=10000):
        self.limit = limit
        self.window = window
        self.clock = clock
        self.max_keys = max_keys
        self._events = OrderedDict()
        self._lock = threading.Lock()

    def _prune(self, dq, now):
        while dq and dq[0] <= now - self.window:
            dq.popleft()

    def hit(self, key):
        """Record an event. Returns None if allowed, else seconds to wait."""
        now = self.clock()
        with self._lock:
            dq = self._events.get(key)
            if dq is None:
                if len(self._events) >= self.max_keys:
                    self._events.popitem(last=False)
                dq = self._events[key] = deque()
            else:
                self._events.move_to_end(key)
            self._prune(dq, now)
            if len(dq) >= self.limit:
                return max(self.window - (now - dq[0]), 0.1)
            dq.append(now)
            return None

    def peek(self, key):
        """Seconds to wait if the key is over its limit, without recording."""
        now = self.clock()
        with self._lock:
            dq = self._events.get(key)
            if dq is None:
                return None
            self._prune(dq, now)
            if len(dq) >= self.limit:
                return max(self.window - (now - dq[0]), 0.1)
            return None


class ReplayCache:
    """Remembers accepted signatures for `ttl` seconds (>= the validity window
    of a timestamp), bounded in size."""

    def __init__(self, ttl, clock=time.monotonic, max_entries=200000):
        self.ttl = ttl
        self.clock = clock
        self.max_entries = max_entries
        self._seen = OrderedDict()
        self._lock = threading.Lock()

    def add(self, token):
        """True if `token` is new (and now remembered), False if it is a replay."""
        now = self.clock()
        with self._lock:
            while self._seen:
                oldest_token, expires = next(iter(self._seen.items()))
                if expires <= now or len(self._seen) >= self.max_entries:
                    self._seen.popitem(last=False)
                else:
                    break
            if token in self._seen:
                return False
            self._seen[token] = now + self.ttl
            return True


class Authenticator:

    def __init__(self, settings, clock=time.time):
        self.settings = settings
        self.clock = clock
        self.replay = ReplayCache(settings.replay_ttl_s)
        self.key_limit = SlidingWindow(settings.rate_limit_per_min)
        self.fail_limit = SlidingWindow(settings.auth_fail_limit_per_min)
        self._key_digests = [(k, _digest(k.key)) for k in settings.keys]

    # -- client address ----------------------------------------------------
    def client_ip(self, peer, forwarded_for):
        """The peer address, or - when the peer is a trusted proxy - the right
        most address of X-Forwarded-For that is not itself a trusted proxy."""
        s = self.settings
        if s.trusted_proxies and peer and ip_in(peer, s.trusted_proxies) and forwarded_for:
            hops = [h.strip() for h in forwarded_for.split(',') if h.strip()]
            for hop in reversed(hops):
                if not ip_in(hop, s.trusted_proxies):
                    return hop
        return peer or ''

    # -- the checks ---------------------------------------------------------
    def check_ip(self, ip):
        nets = self.settings.allowed_networks
        if nets and not ip_in(ip, nets):
            raise ApiError(403, 'FORBIDDEN_IP', 'this address is not allowed')

    def authenticate(self, ip, authorization, timestamp, signature, body):
        """Validate one request; returns the matching ApiKey. Raises ApiError.
        Rate limits both failed authentications (per IP) and valid requests
        (per key)."""
        wait = self.fail_limit.peek(ip)
        if wait is not None:
            raise ApiError(429, 'RATE_LIMITED', 'too many failed attempts', retry_after=int(wait) + 1)
        try:
            key = self._check_credentials(authorization, timestamp, signature, body)
        except ApiError:
            self.fail_limit.hit(ip)
            raise
        wait = self.key_limit.hit(key.slot)
        if wait is not None:
            raise ApiError(429, 'RATE_LIMITED', 'rate limit exceeded', retry_after=int(wait) + 1)
        # one-time use of the signature, only once everything else is fine so
        # a rate limited (not processed) request can be retried unchanged
        if not self.replay.add((key.slot, (signature or '').strip().lower())):
            raise ApiError(401, 'REPLAYED_REQUEST', 'this signed request was already used')
        return key

    def _check_credentials(self, authorization, timestamp, signature, body):
        token = ''
        if authorization and authorization[:7].lower() == 'bearer ':
            token = authorization[7:].strip()
        if not token:
            raise ApiError(401, 'UNAUTHORIZED', 'missing or invalid API key')
        token_digest = _digest(token)
        matched = None
        for key, digest in self._key_digests:           # no early exit: constant work
            if hmac.compare_digest(token_digest, digest):
                matched = key
        if matched is None:
            raise ApiError(401, 'UNAUTHORIZED', 'missing or invalid API key')

        try:
            ts = float((timestamp or '').strip())
        except ValueError:
            ts = None
        if ts is None or ts != ts or ts in (float('inf'), float('-inf')):
            raise ApiError(401, 'STALE_TIMESTAMP', 'missing or invalid X-Timestamp')
        if abs(self.clock() - ts) > self.settings.ts_tolerance_s:
            raise ApiError(401, 'STALE_TIMESTAMP', 'timestamp outside the allowed window')

        expected = sign(matched.secret, timestamp.strip(), body)
        if not signature or not hmac.compare_digest(signature.strip().lower().encode('utf-8'),
                                                    expected.encode('utf-8')):
            raise ApiError(401, 'BAD_SIGNATURE', 'signature does not match')
        return matched
