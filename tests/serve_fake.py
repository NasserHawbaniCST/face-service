# -*- coding: utf-8 -*-
"""Start the REAL FastAPI app (real auth, limits, pool, JSON shapes) with the
deterministic fake engine - used by the gateway addon's end-to-end test:

    python tests/serve_fake.py --port 8766 --key <32+ chars> --secret <32+ chars>
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import uvicorn  # noqa: E402

from app.config import load_settings  # noqa: E402
from app.main import create_app  # noqa: E402
from fake_engine import FakeEngine  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--key', required=True)
    parser.add_argument('--secret', required=True)
    parser.add_argument('--workers', default='2')
    args = parser.parse_args()
    settings = load_settings({
        'FACE_API_KEY_1': args.key, 'FACE_HMAC_SECRET_1': args.secret,
        'FACE_WORKERS': args.workers, 'FACE_RATE_LIMIT_PER_MIN': '1000',
        'FACE_LOG_LEVEL': 'WARNING'})
    app = create_app(settings, engine_factory=lambda: FakeEngine())
    uvicorn.run(app, host='127.0.0.1', port=args.port, log_level='warning', access_log=False)


if __name__ == '__main__':
    main()
