# Image Registry + Verification API (Weeks 1-2)

Backend for the image verification project. Every uploaded image is fingerprinted with a SHA-256 hash
(exact match) and a perceptual hash (near-duplicate match), stored in PostgreSQL, and served through a
FastAPI REST API. Week 2 adds `POST /verify`, which looks an image up in the registry and runs it through
the three verification barriers (watermark, deepfake, tampering).

## Setup

### 1. PostgreSQL 14 or newer
Version 14+ is required because similarity search uses SQL `bit_count`.
Create two databases (the second is used only by the tests):

```bash
psql -U postgres -c "CREATE DATABASE image_registry;"
psql -U postgres -c "CREATE DATABASE image_registry_test;"
```

### 2. Python environment
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure
```bash
cp .env.example .env             # Windows: copy .env.example .env
```
Edit `DATABASE_URL` and `TEST_DATABASE_URL` in `.env` so the user and password match your PostgreSQL install.

### 4. Create the tables
```bash
alembic upgrade head
```

### 5. Run the server
```bash
uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000/docs for interactive API docs (you can upload images from the browser there).

### 6. Run the tests
```bash
pytest
```
Tests use `TEST_DATABASE_URL` and roll back after every test, so real data is never touched.
The barrier pipeline is replaced by a fake in the tests, so no real models are needed.

## API

| Method | Path | Description | Success |
|---|---|---|---|
| POST | `/images` | Upload a JPEG/PNG (multipart field `file`) and **register** it | 201 new, 200 duplicate |
| GET | `/images/{id}` | Image metadata | 200 |
| GET | `/images/{id}/file` | The stored image file | 200 |
| GET | `/images?limit=50&offset=0` | List registered images (newest first) | 200 |
| POST | `/verify` | Upload a JPEG/PNG (multipart field `file`); registry lookup + barriers; saves a verification record. **Does not register the image.** | 200 |
| GET | `/verifications/{id}` | A saved verification record | 200 |
| GET | `/verifications?limit=50&offset=0` | List saved verifications (newest first) | 200 |
| GET | `/health` | Liveness check | 200 |

Errors always look like `{"error": {"code": "...", "message": "..."}}`:

| Status | When |
|---|---|
| 400 | empty file, corrupt/truncated JPEG or PNG, invalid parameters, missing `file` field |
| 404 | unknown image or verification id, unknown route, stored file missing |
| 413 | file larger than `MAX_UPLOAD_BYTES` (default 10 MB) |
| 415 | not a JPEG or PNG (decided from file contents, not the filename or Content-Type) |
| 500 | unexpected server error (`internal_error`); details go to the server log, never to the client |

`POST /verify` uses exactly the same validation as `POST /images`, so its 400/413/415 errors are identical.

Try it:
```bash
curl -F file=@photo.png http://127.0.0.1:8000/images
curl -F file=@photo.png http://127.0.0.1:8000/verify
curl http://127.0.0.1:8000/verifications/1
```
`phash` is returned as a 16-character hex string (a 64-bit number can lose precision as a JSON number).

### `POST /verify` response example

```json
{
  "status": "success",
  "request_id": 12,
  "registry": {"found": true, "image_id": 7, "match_type": "similar", "distance": 2},
  "barriers_results": {
    "watermark": {"status": "ok", "result": {"barrier": "watermark", "status": "intact", "payload_matched": true}},
    "deepfake":  {"status": "ok", "result": {"barrier": "deepfake", "manipulation_score": 0.05}},
    "tampering": {"status": "error", "result": null, "error": "model file not found"}
  },
  "verdict": null,
  "confidence_score": null,
  "message": "Some barriers failed (tampering); the others completed. Verdict is not available yet: the fusion engine is not connected."
}
```

- `status`: `success` (all barriers ok), `partial` (at least one failed), `failed` (all failed). A crashing
  barrier never fails the request; it shows up as `{"status": "error", "result": null, "error": "..."}`.
- `registry.match_type`: `exact` (same bytes, distance 0), `similar` (closest near-duplicate within the Hamming
  threshold), `none` (`image_id` and `distance` are `null`).
- `barriers_results.<name>.result` is whatever the barrier's `process()` returned (made JSON-safe: numpy values
  become plain lists/numbers, NaN becomes `null`).

### Decision: `/verify` is read-only for the registry

`POST /verify` looks the image up but never registers it. `POST /images` is the only way to add an image to the
registry. Reasons: verifying an image should not change what counts as "known", otherwise a tampered image
that is verified once would match itself as a known image forever. Each `/verify` call only writes a row to
`verification_requests` (with `image_id` set to the matched registry image, if any, and `NULL` if the image is
later deleted).

## For Person 1: plugging in the fusion engine

`verdict` (string) and `confidence_score` (float) are in the response but are always `null` for now.
Compute them in **one place**: `app/services/verification.py`, function `verify_image`, at the block marked
`WEEK 3 - FUSION ENGINE HOOK`. At that point you have:

- `barriers_results`: `{"watermark": {...}, "deepfake": {...}, "tampering": {...}}`, where each entry is
  `{"status": "ok", "result": <your barrier dict>}` or `{"status": "error", "result": None, "error": "..."}`.
  Handle the error case (a barrier may have failed).
- `registry`: `{"found", "image_id", "match_type", "distance"}`, useful as an extra signal.

Set `verdict` and `confidence_score` there. They are saved with the record and returned by `POST /verify` and
`GET /verifications/{id}`; no schema, migration or endpoint change is needed.

The barrier interface is unchanged: this code only calls `pipeline.watermark_barrier.process(path)`,
`pipeline.deepfake_barrier.process(path)` and `pipeline.tampering_barrier.process(path)`, where `path` is a
temporary `.jpg`/`.png` file that is deleted after the request. `run_pipeline` is not used.

## For the pipeline owner: direct registry lookup

```python
from app.db.session import get_session_factory
from app.registry.lookup import lookup_image

with get_session_factory()() as db:
    result = lookup_image(db, image_bytes)      # optional: threshold=...
print(result.to_dict())
# {"found": True, "image_id": 7, "match_type": "similar", "distance": 4}
```

## Layout

```
app/
  main.py              the single FastAPI app
  config.py            settings from environment / .env
  api/                 routes.py (images), verify_routes.py (verification), validation, errors, schemas
  registry/            hashing.py, service.py (register/find), lookup.py (registry lookup)
  services/            pipeline.py (Person 1), verification.py (registry + barriers + saved record)
  schema/              verification_schema.py (Person 1)
  storage/             saves image files (named by hash, never by user filename)
  db/                  SQLAlchemy base, models (Image, VerificationRequest), session
alembic/versions/      0001 images table, 0002 verification_requests table
tests/                 hashing, registry, API, lookup, verify tests
```

## Design notes
- **pHash storage:** pHash is an unsigned 64-bit value but PostgreSQL BIGINT is signed. `to_signed` /
  `from_signed` in `hashing.py` convert on write/read; the bit pattern is preserved so SQL and Python agree.
- **Similarity in SQL:** `bit_count((phash # :q)::bit(64))` computes Hamming distance inside the database.
  It scans all rows, which is fine at this scale.
- **Threshold:** `HAMMING_THRESHOLD` (default 10 of 64 bits) decides what counts as "similar". Tune it on real images.
- **Duplicates:** the same file bytes (same SHA-256) return the existing record and store nothing new.
