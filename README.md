# Image Registry (Week 1)

Backend image registry and API for the image verification project. Every uploaded image is
fingerprinted with a SHA-256 hash (exact match) and a perceptual hash (near-duplicate match),
stored in PostgreSQL, and served through a FastAPI REST API.

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

## API

| Method | Path | Description | Success |
|---|---|---|---|
| POST | `/images` | Upload a JPEG/PNG (multipart field `file`) | 201 new, 200 duplicate |
| GET | `/images/{id}` | Image metadata | 200 |
| GET | `/images/{id}/file` | The stored image file | 200 |
| GET | `/images?limit=50&offset=0` | List registered images (newest first) | 200 |
| GET | `/health` | Liveness check | 200 |

Errors always look like `{"error": {"code": "...", "message": "..."}}`:

| Status | When |
|---|---|
| 400 | empty file, corrupt/truncated JPEG or PNG, invalid parameters, missing `file` field |
| 404 | unknown image id, unknown route, stored file missing |
| 413 | file larger than `MAX_UPLOAD_BYTES` (default 10 MB) |
| 415 | not a JPEG or PNG (decided from file contents, not the filename or Content-Type) |

Try it:
```bash
curl -F file=@photo.png http://127.0.0.1:8000/images
curl http://127.0.0.1:8000/images/1
curl -o out.png http://127.0.0.1:8000/images/1/file
```
`phash` is returned as a 16-character hex string (a 64-bit number can lose precision as a JSON number).

## For the pipeline owner (Week 3)

```python
from app.db.session import get_session_factory
from app.registry.lookup import lookup_image

with get_session_factory()() as db:
    result = lookup_image(db, image_bytes)      # optional: threshold=...
print(result.to_dict())
# {"found": True, "image_id": 7, "match_type": "similar", "distance": 4}
```
`match_type` is `"exact"` (same bytes, distance 0), `"similar"` (closest near-duplicate within the
Hamming threshold), or `"none"` (`image_id` and `distance` are `None`).
`lookup_hashes(db, sha256, phash)` does the same if you already have the hashes.

## Layout

```
app/
  main.py            FastAPI app
  config.py          settings from environment / .env
  api/               routes, validation, error handling, response schemas
  registry/          hashing.py, service.py (register/find), lookup.py (pipeline lookup)
  storage/           saves image files (named by hash, never by user filename)
  db/                SQLAlchemy base, model, session
alembic/             migrations
tests/               hashing, registry, API, lookup tests
```

## Design notes
- **pHash storage:** pHash is an unsigned 64-bit value but PostgreSQL BIGINT is signed. `to_signed` /
  `from_signed` in `hashing.py` convert on write/read; the bit pattern is preserved so SQL and Python agree.
- **Similarity in SQL:** `bit_count((phash # :q)::bit(64))` computes Hamming distance inside the database.
  It scans all rows, which is fine at this scale.
- **Threshold:** `HAMMING_THRESHOLD` (default 10 of 64 bits) decides what counts as "similar". Tune it on real images.
- **Duplicates:** the same file bytes (same SHA-256) return the existing record and store nothing new.
