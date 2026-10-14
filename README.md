# face_service - stateless face-processing microservice

OpenCV **YuNet** (face detector, MIT) + **SFace** (face recogniser, Apache-2.0) behind a small
authenticated HTTPS API (FastAPI + uvicorn). It is the optional **external engine** of the Identity
Guard (`ai_addons/cs_identity_gateway`): the gateway sends the selfie frames of ONE verification, the
service answers with numbers (face count, box, 5 landmarks, yaw, quality, 128-d embedding). The
gateway compares with the employee's enrolled template and takes the decision.

Languages: **English** first, then **العربية** (second half of this file).

---

## 0. Deployment cheat sheet

**A. On the face server** (fresh Ubuntu 22.04 / 24.04, 2-4 vCPU, 2 GB RAM, ideally inside KSA)

```bash
sudo apt update && sudo apt install -y docker.io docker-compose-v2 ufw
# copy this folder to /opt/face_service  (scp -r face_service user@server:/opt/)
cd /opt/face_service && cp .env.example .env && chmod 600 .env
python3 -m app.genkeys            # or: openssl rand -hex 32  (twice) -> paste into .env:
#   FACE_API_KEY_1=...   FACE_HMAC_SECRET_1=...
#   FACE_ALLOWED_IPS=<gateway public IP>     FACE_WORKERS=<number of vCPUs, max 4>
#   FACE_DOMAIN=face.yourdomain.com          FACE_TRUSTED_PROXIES=172.16.0.0/12
# DNS: face.yourdomain.com -> this server's IP, then:
sudo docker compose --profile tls up -d --build          # service + Caddy (automatic HTTPS)
sudo ufw default deny incoming && sudo ufw allow OpenSSH && sudo ufw allow 80/tcp
sudo ufw allow from <GATEWAY_IP> to any port 443 proto tcp && sudo ufw enable
curl https://face.yourdomain.com/health                  # -> {"status":"ok","engine_version":"yunet_2023mar+sface_2021dec",...}
```

**B. On the gateway (Odoo)**

1. Upgrade the module `cs_identity_gateway` once (adds the settings page).
2. Settings > **Identity Gateway** > Face engine: *Face engine* = **External face service**,
   *Face service URL* = `https://face.yourdomain.com`, *API key* / *HMAC secret* = the values from step A.
3. Click **Test connection** - it must answer "Connected. Engine yunet_2023mar+sface_2021dec ... health N ms,
   analyse N ms". Then **Save**.
4. Rollback at any time: set *Face engine* back to **Local** - nothing else changes.

Without a domain / Caddy (private network between gateway and service): `docker compose up -d --build`,
set `FACE_BIND=<private IP of the server>` and use `http://<private IP>:8765` as URL (TLS is strongly
recommended, see section 5).

---

## 1. Why this exists and how it fits

```
phone ──(JSON-RPC, unchanged)──> GATEWAY (Odoo) ──signed HTTPS──> FACE SERVICE (this project)
                                   │  auth, challenge, policy, tokens       detect / landmarks / yaw /
                                   │  fetches the enrolled template         quality / embedding only
                                   │  from the CUSTOMER's Odoo              no state, no credentials
                                   └─ compares, decides, issues the token
```

* **Why not phone -> face service directly?** The gateway holds the sessions, the challenge (random
  liveness steps, TTL 90 s, single use), the subscription policy, the attempt counters / lock-out and the
  token that the attendance routes require. If the phone talked to the face service, all of that would
  have to be duplicated there, and the service would have to be trusted with decisions. Decision taken:
  the service is a dumb, stateless "pixels in, numbers out" function; only the gateway talks to it.
* The service **never receives** customer credentials, sessions, user ids, device ids or the enrolled
  template, and **never persists** anything (read-only container, no database, no disk writes).
* The REST contract between the phone and the gateway is **unchanged** (`docs/IDENTITY_GUARD_CONTRACT.md`).
* The same two model files (sha256 listed below) and the same OpenCV logic as the gateway's local engine
  (`app/engine.py` mirrors `cs_identity_gateway/services/face_engine.py`; a gateway test compares them).

| model file | sha256 | licence |
|---|---|---|
| `models/face_detection_yunet_2023mar.onnx` | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` | MIT |
| `models/face_recognition_sface_2021dec.onnx` | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` | Apache-2.0 |

The hashes are verified at start-up; a modified model file makes the service refuse to start the engine.

---

## 2. API

### `GET /health` (no authentication)

```json
{"status": "ok", "service_version": "1.0.0", "engine_version": "yunet_2023mar+sface_2021dec",
 "models": {"detector": {"file": "face_detection_yunet_2023mar.onnx", "sha256": "8f23..."},
            "recognizer": {"file": "face_recognition_sface_2021dec.onnx", "sha256": "0ba9..."}},
 "workers": 2}
```
HTTP 503 with `"status": "unavailable"` and `"error"` when the models could not be loaded.

### `POST /v1/analyze`

Headers (all required):

| header | value |
|---|---|
| `Authorization` | `Bearer <API key>` |
| `X-Timestamp` | unix time in seconds, fractions allowed (`1790000000.123`); +-60 s tolerance |
| `X-Signature` | `sha256=` + hex `HMAC-SHA256(secret, timestamp + "." + raw_body)` |
| `X-Request-Id` | optional, echoed back (a UUID is generated when missing / invalid) |

Body (`application/json`, at most 4 MiB, at most 6 frames, each frame at most 400 KB decoded JPEG):

```json
{"request_id": "b17253d0c41a41ea9cf209b91df89187",
 "frames": [{"id": "0", "image_b64": "<base64 of a JPEG>"},
            {"id": "1", "image_b64": "..."}]}
```

Response `200`:

```json
{"request_id": "b17253d0...", "engine_version": "yunet_2023mar+sface_2021dec",
 "frames": [
  {"id": "0", "ok": true, "faces": 1, "score": 0.9931,
   "box": [112.4, 80.1, 160.0, 205.3],
   "landmarks": [[150.2,150.0],[230.1,148.9],[191.0,190.2],[158.7,235.0],[225.5,234.1]],
   "yaw": 0.0312, "area_ratio": 0.1426, "quality": 0.62,
   "width": 480, "height": 640, "embedding": [0.0123, -0.0456, "... 128 numbers"]},
  {"id": "1", "ok": true, "faces": 0, "score": null, "box": null, "landmarks": null, "yaw": null,
   "area_ratio": null, "quality": null, "width": 480, "height": 640, "embedding": null},
  {"id": "2", "ok": false, "error": "bad_image", "faces": 0, "...": null, "width": 0, "height": 0}
 ]}
```

Semantics (identical to the gateway's local engine):

* An image must be a **JPEG**; its size is read from the header and refused **before decoding** when
  larger than 8 megapixels (decompression bombs). Images are downscaled so the long edge is <= 640 px
  (`width` / `height` are the analysed size; `box` and `landmarks` are in those coordinates).
* `faces` = number of faces. `score`, `box` `[x, y, w, h]`, `landmarks` (5 points: right eye, left eye,
  nose tip, right mouth corner, left mouth corner), `yaw`, `area_ratio`, `quality` describe the most
  confident face (`null` when there is none).
* `yaw` = (nose x - eye-midpoint x) / inter-eye distance: ~0 frontal, the sign tells the turn direction.
* `area_ratio` = box area / (shorter image edge)^2. `quality` = 0..1 sharpness of the face crop
  (Laplacian variance) - informational, the gateway does not decide on it yet.
* `embedding` (128 numbers, SFace) is returned **only when exactly one face** is found.
* Per-frame problems do not fail the request: `ok: false` with `error` = `bad_image` | `bad_base64` |
  `engine_error`.

Request-level errors: `{"error": {"code": "...", "message": "..."}, "request_id": "..."}` with the stable
codes below (and `Retry-After` where relevant):

| HTTP | code | meaning |
|---|---|---|
| 400 | `BAD_REQUEST` | not JSON / wrong shape / duplicate frame ids / bad `request_id` |
| 400 | `TOO_MANY_FRAMES` | more than 6 frames |
| 401 | `UNAUTHORIZED` | missing / wrong API key |
| 401 | `BAD_SIGNATURE` | signature does not match the body / secret |
| 401 | `STALE_TIMESTAMP` | `X-Timestamp` missing, invalid or outside +-60 s (check the clocks, use NTP) |
| 401 | `REPLAYED_REQUEST` | this exact signed request was already accepted |
| 403 | `FORBIDDEN_IP` | address not in `FACE_ALLOWED_IPS` |
| 413 | `PAYLOAD_TOO_LARGE` / `FRAME_TOO_LARGE` | body > `FACE_MAX_BODY_BYTES` / a frame > 400 KB |
| 429 | `RATE_LIMITED` | per-key rate limit, or too many failed authentications from one IP |
| 503 | `ENGINE_UNAVAILABLE` / `BUSY` | models not loaded / queue full (retry shortly) |
| 504 | `TIMEOUT` | analysis exceeded `FACE_REQUEST_TIMEOUT` |
| 500 | `INTERNAL_ERROR` | unexpected (details only in the server log, never the request) |

Signing example (Python):

```python
import hashlib, hmac, json, time, requests
body = json.dumps({"request_id": "demo-1", "frames": [{"id": "0", "image_b64": b64_jpeg}]},
                  separators=(",", ":")).encode()
ts = "%.3f" % time.time()
sig = "sha256=" + hmac.new(SECRET.encode(), ts.encode() + b"." + body, hashlib.sha256).hexdigest()
requests.post(URL + "/v1/analyze", data=body, timeout=(3, 15), headers={
    "Authorization": "Bearer " + API_KEY, "X-Timestamp": ts, "X-Signature": sig,
    "Content-Type": "application/json"})
```
(The signature covers the bytes actually sent: serialise once, sign that, send that.)

### Protections implemented

Bearer key compared in constant time (two keys accepted for rotation) **and** HMAC signature over the
body; timestamp window + one-time signature cache (a captured request cannot be replayed or altered);
optional IP allow-list; per-key rate limit and a per-IP limit on failed authentications; strict body
limit enforced at the ASGI layer (Content-Length check + counted stream, before parsing); decode
guards; CPU work in a thread pool (one engine per thread, the event loop stays free); bounded queue
(`503 BUSY` instead of piling up); **no logging of images, embeddings or bodies** - one line per request:
request id, status, milliseconds, frame count, faces per frame, which key slot.

> The replay cache and the rate limits live in the memory of ONE process. Run a single uvicorn process
> (the default here) and scale with `FACE_WORKERS`. Several processes / replicas behind a load balancer
> would each have their own cache - use sticky routing by source IP if you ever need that.

---

## 3. Configuration (environment variables)

See `.env.example`. Required: `FACE_API_KEY_1`, `FACE_HMAC_SECRET_1` (>= 32 characters each; the service
refuses to start with weaker values).

| variable | default | meaning |
|---|---|---|
| `FACE_API_KEY_1` / `FACE_HMAC_SECRET_1` | - | credential pair |
| `FACE_API_KEY_2` / `FACE_HMAC_SECRET_2` | - | second pair, only while rotating |
| `FACE_HMAC_SECRET` | - | fallback secret for a slot without its own |
| `FACE_ALLOWED_IPS` | empty (all) | comma separated IPs / CIDRs allowed on `/v1/analyze` |
| `FACE_TRUSTED_PROXIES` | empty | proxies whose `X-Forwarded-For` is trusted (Caddy / nginx) |
| `FACE_WORKERS` | min(CPUs, 4) | engines = parallel analyses |
| `FACE_CV_THREADS` | 1 | OpenCV threads per engine |
| `FACE_MAX_QUEUE` | 2 x workers | waiting requests before `503 BUSY` |
| `FACE_REQUEST_TIMEOUT` | 20 | seconds |
| `FACE_RATE_LIMIT_PER_MIN` | 120 | per API key |
| `FACE_AUTH_FAIL_LIMIT_PER_MIN` | 30 | failed authentications per IP |
| `FACE_MAX_BODY_BYTES` | 4194304 | request body limit |
| `FACE_MAX_FRAMES` / `FACE_MAX_FRAME_BYTES` | 6 / 409600 | frames per request / decoded size |
| `FACE_TS_TOLERANCE` | 60 | seconds |
| `FACE_MODELS_DIR` | `./models` | where the two ONNX files are |
| `FACE_LOG_LEVEL` | INFO | |

---

## 4. Install on a fresh Ubuntu server

### 4a. Docker (recommended)

```bash
sudo apt update && sudo apt install -y docker.io docker-compose-v2 ufw
sudo mkdir -p /opt/face_service && sudo chown $USER /opt/face_service
# from your PC:  scp -r face_service/* user@SERVER:/opt/face_service/   (includes models/*.onnx)
cd /opt/face_service
cp .env.example .env && chmod 600 .env && nano .env          # keys, FACE_ALLOWED_IPS, FACE_DOMAIN
sudo docker compose up -d --build                             # without TLS front (private network)
# or with automatic HTTPS (needs DNS + ports 80/443):
sudo docker compose --profile tls up -d --build
sudo docker compose ps ; curl -s http://127.0.0.1:8765/health
```
The container runs as a non-root user, read-only file system, all capabilities dropped, memory limit 1 GB,
health check on `/health`, rotated json logs. Update = copy the new files, `docker compose up -d --build`.

> Docker publishes ports around `ufw`. Keep `FACE_BIND=127.0.0.1` (the default) and let Caddy / nginx be
> the only public entry, and rely on `FACE_ALLOWED_IPS` (+ `FACE_TRUSTED_PROXIES`) as the real access
> control. (The Docker image could not be built in the environment where this project was written - the
> first `docker compose build` on your server is its first real run; fix anything it reports.)

### 4b. Without Docker (systemd)

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip ufw caddy
sudo useradd --system --no-create-home --shell /usr/sbin/nologin face
sudo mkdir -p /opt/face_service && sudo cp -r face_service/* /opt/face_service/
cd /opt/face_service && sudo python3 -m venv venv && sudo venv/bin/pip install -r requirements.txt
sudo install -m 640 -o root -g face .env.example /etc/face-service.env && sudo nano /etc/face-service.env
sudo cp deploy/face-service.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now face-service
sudo systemctl status face-service ; curl -s http://127.0.0.1:8765/health
```
Caddy in front (`/etc/caddy/Caddyfile`):
```
face.yourdomain.com {
    request_body { max_size 5MB }
    reverse_proxy 127.0.0.1:8765
}
```
`sudo systemctl reload caddy` (certificate is issued automatically). Set `FACE_TRUSTED_PROXIES=127.0.0.1`
in `/etc/face-service.env` so `FACE_ALLOWED_IPS` sees the real client address. (nginx works as well:
`proxy_pass http://127.0.0.1:8765; proxy_set_header X-Forwarded-For $remote_addr; client_max_body_size 5m;`.)

---

## 5. Network security

* **TLS is mandatory** on any path that crosses the internet (the frames are biometric data). Use the
  Caddy profile (automatic Let's Encrypt) or nginx + certbot. Plain `http://` is acceptable only on a
  private network between the two servers (VPN / VPC).
* **Firewall** - allow SSH, port 80 (Let's Encrypt HTTP challenge + redirect) and port 443 **only from the
  gateway's IP**:
  ```bash
  sudo ufw default deny incoming && sudo ufw allow OpenSSH && sudo ufw allow 80/tcp
  sudo ufw allow from <GATEWAY_IP> to any port 443 proto tcp && sudo ufw enable
  ```
  and `FACE_ALLOWED_IPS=<GATEWAY_IP>` as a second layer. The service cannot be used without the key AND
  the HMAC secret AND a fresh timestamp, but there is no reason to expose it to the world.
* Never put the keys in git; `.env` is mode 600 / `/etc/face-service.env` mode 640 root:face.
* Keep the server clock right (`timedatectl set-ntp true`): the +-60 s timestamp window needs it, on
  both servers.

### Key rotation (no downtime)

1. `python3 -m app.genkeys` -> new pair.
2. On the service set `FACE_API_KEY_2` / `FACE_HMAC_SECRET_2` to the new pair (keep slot 1), restart
   (`docker compose up -d` / `systemctl restart face-service`). Both pairs are accepted now.
3. Gateway: Settings > Identity Gateway: enter the new key / secret, **Test connection**, Save.
4. Watch the service log: requests show `key=2`. Then move slot 2 into slot 1 (or just delete slot 1)
   and restart. Do this every few months and whenever someone with access leaves.

---

## 6. Capacity sizing (CPU only, no GPU)

* One frame = JPEG decode + YuNet detection (+ SFace embedding when exactly one face) = roughly
  **50-100 ms of one CPU core** on a normal cloud vCPU (measured here on a fast desktop CPU, blank
  480x640 frames: ~40 ms per frame including HTTP; real faces add the embedding step - **real-face
  latency was not measured**, there were no face photos).
* One verification / enrolment = 3 frames = **~150-300 ms of CPU**, one HTTPS round trip.
* Throughput ~ `FACE_WORKERS` / 0.25 s -> **2 workers ~ 8 verifications per second**, 4 workers ~ 16.
  Measured on a 24-thread desktop with blank frames: 5.8 requests/s with one client, 14.8 with two
  (`FACE_WORKERS=2`), saturating there (queue limit then answers `503 BUSY`).
* Rule of thumb: **1 vCPU per `FACE_WORKERS`, `FACE_WORKERS` <= 4** (set `FACE_CV_THREADS=1`). Example:
  1000 employees checking in within 15 minutes = ~2 verifications/s average, ~6/s peak -> **2-4 vCPU**.
  Need more? Increase vCPUs (`FACE_WORKERS` up to 4 is plenty for this use case) - keep one process.
* Memory: ~180 MB measured with 2 engines after warm-up; plan ~150 MB + ~40 MB per worker, so **>= 512 MB
  free, 1-2 GB machine is comfortable** (the compose file caps the container at 1 GB).
* Gateway side: connect timeout 3 s, read timeout `engine_timeout` (15 s default), one retry on connection
  errors only, circuit breaker (3 consecutive failures -> 20 s of instant `FACE_ENGINE_UNAVAILABLE`).

---

## 7. Privacy and data residency (not legal advice)

* Face images are **biometric personal data** (sensitive under the Saudi PDPL). In this architecture
  frames **leave the gateway** and reach this server: they are decoded and analysed in RAM, then
  discarded when the request ends. Nothing is written to disk or logged; the service stores no
  templates. Only the 128-number template is stored - in the customer's own Odoo database, by the
  gateway.
* Host the service **inside the Kingdom** (a KSA cloud region / data centre) and, ideally, in the same
  jurisdiction and provider class as the gateway. A server outside KSA is a cross-border transfer of
  biometric data and needs a lawful basis / the PDPL transfer conditions; avoid it unless the customer's
  data-protection officer has approved.
* Sign a data-processing agreement with the hosting provider; restrict who has SSH / root; keep OS
  patched; do not enable provider-level disk snapshots / request capture / APM that could record bodies.
* The customer (employer) remains the data controller: privacy notice and consent as described in the
  gateway README.

---

## 8. Monitoring

* `GET /health` (200 / 503) for the load balancer, `docker compose ps` (healthy) or an uptime checker.
* Logs (one line per request, no payloads):
  `rid=... POST /v1/analyze status=200 ms=129 ip=10.0.0.5 key=1 req=... frames=3 faces=1,1,1 ok=3`.
  Watch for `status=401` bursts (wrong key / clock drift / probing), `429`, `503 BUSY` (add workers /
  vCPUs), `504`, and `face engine ready` at start-up. `docker compose logs -f face` /
  `journalctl -u face-service -f`.
* In Odoo (gateway) the same events appear as `FACE_ENGINE_UNAVAILABLE` and log lines
  `identity: remote face engine failed ...` / `identity: analysed N frames with the remote engine in X ms`.
* Settings > Identity Gateway > **Test connection** shows the engine version and latency at any time.

---

## 9. Development / tests

```bash
python -m venv venv && venv/bin/pip install -r requirements-dev.txt
venv/bin/python -m pytest -q                 # auth, limits, API contract, real OpenCV models on blank images
venv/bin/python -m app.genkeys
FACE_API_KEY_1=... FACE_HMAC_SECRET_1=... venv/bin/uvicorn --factory app.main:create_app --port 8765 --no-access-log
```
`tests/serve_fake.py` starts the real app with a deterministic fake engine (used by the gateway's
end-to-end test). `app/engine.py` must stay in step with the gateway's `services/face_engine.py`
(a gateway test `TestServiceParity` loads both and compares them).

---
---

<div dir="rtl">

# خدمة معالجة الوجه (face_service) - النسخة العربية

خدمة مستقلة بلا حالة (stateless) تشغّل نماذج OpenCV: **YuNet** لكشف الوجه و**SFace** لاستخراج البصمة
(128 رقماً) خلف واجهة HTTPS موثّقة. هي «المحرك الخارجي» الاختياري لـ Identity Guard
(`ai_addons/cs_identity_gateway`): البوابة (gateway) ترسل إطارات صورة التحقق (3 إطارات) فتعيد الخدمة
أرقاماً فقط (عدد الوجوه، الصندوق، 5 نقاط، زاوية الدوران، الجودة، البصمة)، والبوابة هي التي تقارن
بالقالب المسجّل وتتخذ القرار.

## 0. ملخص النشر السريع

1. الخادم: Ubuntu 22.04/24.04، من 2 إلى 4 vCPU وذاكرة 2 جيجا، ويُفضّل داخل المملكة.
2. `sudo apt update && sudo apt install -y docker.io docker-compose-v2 ufw`
3. انسخ المجلد إلى `/opt/face_service` ثم: `cp .env.example .env && chmod 600 .env`
4. ولّد المفاتيح: `python3 -m app.genkeys` (أو `openssl rand -hex 32` مرتين) وضعها في `.env`
   (`FACE_API_KEY_1`, `FACE_HMAC_SECRET_1`)، واضبط `FACE_ALLOWED_IPS` على IP البوابة العام،
   و`FACE_WORKERS` على عدد الأنوية (حد أقصى 4).
5. وجّه اسم النطاق (DNS) إلى الخادم واضبط `FACE_DOMAIN` و`FACE_TRUSTED_PROXIES=172.16.0.0/12`.
6. `sudo docker compose --profile tls up -d --build` (الخدمة + Caddy بشهادة HTTPS تلقائية).
7. الجدار الناري: `sudo ufw default deny incoming && sudo ufw allow OpenSSH && sudo ufw allow 80/tcp &&
   sudo ufw allow from <IP البوابة> to any port 443 proto tcp && sudo ufw enable`
8. افحص: `curl https://face.yourdomain.com/health` فيجب أن ترى `"status":"ok"`.
9. في أودو (البوابة): الإعدادات > Identity Gateway > Face engine: اختر «External face service»، ضع الرابط
   والمفتاح وسرّ التوقيع، واضغط **Test connection** (يعرض إصدار المحرك وزمن الاستجابة) ثم احفظ.
10. حدّث الموديول `cs_identity_gateway` مرة واحدة على البوابة. للتراجع: أعد الخيار إلى «Local».

## 1. لماذا بهذا الشكل؟

الهاتف **لا** يتصل بخدمة الوجه مباشرة. البوابة تملك الجلسات والتحدّي العشوائي (3 خطوات، صلاحية 90 ثانية،
استخدام واحد) وسياسة الاشتراك وعدّاد المحاولات والإقفال والتوكن المطلوب في مسارات الحضور؛ نقل ذلك إلى
الخدمة سيكرّره ويجعلها جهة قرار. لذلك الخدمة مجرّد «صورة تدخل، أرقام تخرج»، والبوابة وحدها تتصل بها.
الخدمة لا تستلم أي بيانات اعتماد للعميل ولا جلسات ولا معرّفات مستخدمين أو أجهزة ولا القالب المسجّل،
ولا تحفظ أي شيء (حاوية للقراءة فقط، بلا قاعدة بيانات). واجهة الهاتف مع البوابة لم تتغير.

## 2. الواجهة (API)

* `GET /health` بلا مصادقة: الحالة وإصدار المحرك وبصمات sha256 للنموذجين.
* `POST /v1/analyze` بالترويسات: `Authorization: Bearer <المفتاح>`، `X-Timestamp` (ثوانٍ يونكس، سماحية ±60
  ثانية)، `X-Signature: sha256=` + HMAC-SHA256 للسرّ على النص `timestamp + "." + الجسم الخام`.
  الجسم: `{request_id, frames:[{id, image_b64}]}` حتى 6 إطارات، كل إطار JPEG لا يتجاوز 400 كيلوبايت.
  الرد: `{request_id, engine_version, frames:[{id, ok, error?, faces, box, landmarks, yaw, area_ratio,
  quality, width, height, embedding}]}`. البصمة `embedding` تُعاد **فقط عند وجود وجه واحد بالضبط**.
* رموز الأخطاء الثابتة: `BAD_REQUEST`, `TOO_MANY_FRAMES`, `UNAUTHORIZED`, `BAD_SIGNATURE`,
  `STALE_TIMESTAMP`, `REPLAYED_REQUEST`, `FORBIDDEN_IP`, `PAYLOAD_TOO_LARGE`, `FRAME_TOO_LARGE`,
  `RATE_LIMITED`, `ENGINE_UNAVAILABLE`, `BUSY`, `TIMEOUT`, `INTERNAL_ERROR` (الجدول الإنجليزي أعلاه).
* الحماية: مقارنة المفتاح بزمن ثابت، وتوقيع HMAC على الجسم، ونافذة زمنية مع ذاكرة لمنع إعادة الإرسال،
  وقائمة IP مسموحة اختيارية، وحدّ معدّل لكل مفتاح، وحدّ صارم لحجم الجسم، وفحص أبعاد الصورة قبل فكّها،
  ومعالجة ثقيلة في مسارات (threads) كي تبقى الخدمة مستجيبة، **وبدون تسجيل أي صورة أو بصمة أو جسم طلب**.
* مهم: ذاكرة منع إعادة الإرسال والحدود داخل ذاكرة **عملية واحدة** - شغّل عملية uvicorn واحدة وارفع
  الأداء عبر `FACE_WORKERS`.

## 3. المتغيرات

راجع `.env.example` والجدول الإنجليزي. المطلوب: `FACE_API_KEY_1` و`FACE_HMAC_SECRET_1` (32 حرفاً على الأقل
وإلا ترفض الخدمة التشغيل). مهم أيضاً: `FACE_ALLOWED_IPS`, `FACE_TRUSTED_PROXIES`, `FACE_WORKERS`,
`FACE_RATE_LIMIT_PER_MIN`, `FACE_MAX_BODY_BYTES`, `FACE_MAX_QUEUE`, `FACE_REQUEST_TIMEOUT`.

## 4. التثبيت على Ubuntu جديد

**بـ Docker (الموصى به):** الخطوات 2-6 في الملخص أعلاه. الحاوية تعمل بمستخدم غير root، ونظام ملفات للقراءة
فقط، وبلا صلاحيات إضافية، وحدّ ذاكرة 1 جيجا، وفحص صحة على `/health`.
تنبيه: Docker ينشر المنافذ متجاوزاً `ufw`، فأبقِ `FACE_BIND=127.0.0.1` ودع Caddy/nginx المدخل العام
الوحيد، واعتمد على `FACE_ALLOWED_IPS` كضبط وصول فعلي. (لم يتح بناء صورة Docker في بيئة التطوير؛ أول
`docker compose build` على الخادم هو أول تشغيل حقيقي لها.)

**بدون Docker:** `sudo apt install -y python3-venv python3-pip caddy`، أنشئ مستخدم `face`، انسخ المجلد
إلى `/opt/face_service`، أنشئ venv وثبّت `requirements.txt`، ضع الإعدادات في `/etc/face-service.env`
(صلاحيات 640)، انسخ `deploy/face-service.service` إلى `/etc/systemd/system/` ثم
`systemctl enable --now face-service`، وضع Caddy أمامها (`reverse_proxy 127.0.0.1:8765`) مع
`FACE_TRUSTED_PROXIES=127.0.0.1`.

## 5. الأمان

* **TLS إلزامي** عند عبور الإنترنت (الإطارات بيانات بيومترية). `http://` مقبول فقط على شبكة خاصة بين
  الخادمين.
* الجدار الناري: افتح SSH والمنفذ 80 (لشهادة Let's Encrypt) والمنفذ 443 **من IP البوابة فقط**،
  وضع `FACE_ALLOWED_IPS` كطبقة ثانية. لا تضع المفاتيح في git. اضبط الوقت على الخادمين (NTP).
* **تدوير المفاتيح بدون توقف:** ولّد زوجاً جديداً، ضعه في الخانة 2 (`FACE_API_KEY_2`/`FACE_HMAC_SECRET_2`)
  وأعد تشغيل الخدمة (يُقبل الزوجان)، غيّر المفتاح والسرّ في إعدادات البوابة واضغط Test connection ثم
  احفظ، تأكد في السجل من ظهور `key=2`، ثم احذف الخانة 1 وأعد التشغيل.

## 6. تقدير السعة (CPU فقط)

* الإطار الواحد يحتاج تقريباً **50 إلى 100 ملي ثانية** من نواة واحدة؛ التحقق الواحد = 3 إطارات =
  **150 إلى 300 مللي ثانية** وطلب HTTPS واحد. (القياس عندنا على إطارات فارغة فقط؛ زمن الوجوه الحقيقية
  لم يُقَس لعدم توفر صور وجوه.)
* القاعدة: **نواة واحدة لكل `FACE_WORKERS` وبحد أقصى 4**. مثال: 1000 موظف يسجّلون خلال 15 دقيقة =
  نحو 2 تحقق/ثانية بالمتوسط و6 عند الذروة ← يكفي **2-4 vCPU**.
* الذاكرة: نحو 180 ميجا مع محرّكين، فخطّط لـ 150 ميجا + 40 ميجا لكل عامل؛ جهاز بذاكرة 1-2 جيجا مريح.
* من جهة البوابة: مهلة اتصال 3 ثوانٍ ومهلة قراءة 15 ثانية، ومحاولة إعادة واحدة عند أخطاء الاتصال فقط،
  وقاطع دائرة (بعد 3 إخفاقات متتالية يُرجع `FACE_ENGINE_UNAVAILABLE` فوراً لمدة 20 ثانية).

## 7. الخصوصية وإقامة البيانات (ليست استشارة قانونية)

* صور الوجه بيانات شخصية بيومترية (حساسة في نظام حماية البيانات الشخصية السعودي PDPL). في هذا التصميم
  تغادر الإطارات البوابة وتصل إلى هذا الخادم، فتُفكّ وتُحلَّل في الذاكرة ثم تُهمَل عند انتهاء الطلب؛ لا
  تُكتب على القرص ولا تُسجَّل. القالب (128 رقماً) فقط يُخزَّن في قاعدة بيانات العميل عبر البوابة.
* استضف الخدمة **داخل المملكة** (منطقة سحابية أو مركز بيانات سعودي)؛ الخادم خارج المملكة يُعدّ نقلاً
  عابراً للحدود لبيانات بيومترية ويحتاج أساساً نظامياً وموافقة مسؤول حماية البيانات لدى العميل.
* وقّع اتفاقية معالجة بيانات مع مزوّد الاستضافة، وقيّد صلاحيات SSH، وحدّث النظام، وعطّل أي لقطات أو
  أدوات مراقبة قد تسجّل أجسام الطلبات. العميل (صاحب العمل) يبقى المتحكّم بالبيانات.

## 8. المراقبة

* `GET /health` (200 / 503) لأدوات المراقبة وفحص الحاوية.
* السجل سطر واحد لكل طلب بلا بيانات: المعرّف والحالة والزمن وعدد الإطارات وعدد الوجوه ورقم المفتاح.
  راقب موجات `401` (مفتاح خاطئ/فرق ساعة)، و`429`، و`503 BUSY` (زِد العوامل أو الأنوية)، و`504`.
* في أودو زر **Test connection** يعرض إصدار المحرك والزمن في أي وقت، وتظهر الأعطال كـ
  `FACE_ENGINE_UNAVAILABLE` في البوابة.

</div>
