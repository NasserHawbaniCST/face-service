# -*- coding: utf-8 -*-
"""Print a fresh API key + HMAC secret:  python -m app.genkeys"""
import secrets

if __name__ == '__main__':
    print('FACE_API_KEY_1=%s' % secrets.token_hex(32))
    print('FACE_HMAC_SECRET_1=%s' % secrets.token_hex(32))
