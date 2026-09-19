#!/usr/bin/env python3

"""
 Production Movie Poster Downloader

Source:
https://cdn.jsdelivr.net/gh/unknownman2024/bms-interest-track@main/Bookmyshow%20Data/moviesdb.json

Output:
images/<movie-slug>.jpg

IMPORTANT:
Existing valid posters are NEVER downloaded again.

A poster is downloaded only when:
    1. File does not exist
    2. Existing file is invalid
    3. Existing file is > 20 KB
    4. Poster URL changed
    5. Slug changed / new movie

Designed for 2,000+ movies.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import sys
import tempfile
import time
import unicodedata

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageOps
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ============================================================
# CONFIG
# ============================================================

SOURCE_URL = (
    "https://cdn.jsdelivr.net/gh/unknownman2024/"
    "bms-interest-track@main/Bookmyshow%20Data/moviesdb.json"
)

OUTPUT_DIR = Path("images")

# This file remembers which source URL produced each poster.
MANIFEST_FILE = Path("poster-manifest.json")

MAX_FILE_SIZE = 20 * 1024

# 2,000+ movies:
# 32 is usually a good balance for GitHub Actions.
WORKERS = 32

CONNECT_TIMEOUT = 10
READ_TIMEOUT = 30

MAX_RETRIES = 4
BACKOFF_FACTOR = 0.5

MIN_QUALITY = 25
MAX_QUALITY = 95

RESIZE_FACTOR = 0.90

MIN_WIDTH = 120
MIN_HEIGHT = 180

MAX_PIXELS = 20_000_000

USER_AGENT = (
    "Poster-Downloader/2.0 "
    "(production)"
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("filmy-posters")


# ============================================================
# HTTP SESSION
# ============================================================

def create_session() -> requests.Session:

    session = requests.Session()

    retry = Retry(
        total=MAX_RETRIES,
        connect=MAX_RETRIES,
        read=MAX_RETRIES,
        status=MAX_RETRIES,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        respect_retry_after_header=True,
        raise_on_status=False,
    )

    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=WORKERS,
        pool_maxsize=WORKERS,
    )

    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "application/json,image/*,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    })

    return session


# ============================================================
# SLUG
# ============================================================

def slugify(value: str) -> str:

    value = str(value or "").strip()

    value = unicodedata.normalize(
        "NFKD",
        value,
    )

    value = value.encode(
        "ascii",
        "ignore",
    ).decode("ascii")

    value = value.lower()

    value = value.replace("&", " and ")

    value = value.replace("'", "")

    value = re.sub(
        r"[^a-z0-9]+",
        "-",
        value,
    )

    value = re.sub(
        r"-+",
        "-",
        value,
    )

    return value.strip("-") or "movie"


# ============================================================
# MANIFEST
# ============================================================

def load_manifest() -> dict[str, Any]:

    if not MANIFEST_FILE.exists():
        return {}

    try:

        with MANIFEST_FILE.open(
            "r",
            encoding="utf-8",
        ) as f:

            data = json.load(f)

        if not isinstance(data, dict):
            return {}

        return data

    except Exception as exc:

        logger.warning(
            "Manifest could not be loaded: %s",
            exc,
        )

        return {}


def save_manifest(
    manifest: dict[str, Any],
) -> None:

    temp = MANIFEST_FILE.with_suffix(".tmp")

    temp.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    os.replace(
        temp,
        MANIFEST_FILE,
    )


# ============================================================
# SOURCE DATABASE
# ============================================================

def download_database(
    session: requests.Session,
) -> dict[str, Any]:

    logger.info("Fetching moviesdb.json...")

    response = session.get(
        SOURCE_URL,
        timeout=(
            CONNECT_TIMEOUT,
            READ_TIMEOUT,
        ),
    )

    response.raise_for_status()

    data = response.json()

    if not isinstance(data, dict):
        raise ValueError(
            "moviesdb.json is not an object"
        )

    return data


# ============================================================
# MOVIE NORMALIZATION
# ============================================================

def normalize_movies(
    data: dict[str, Any],
) -> list[dict[str, Any]]:

    movies = []

    for movie_code, movie in data.items():

        if not isinstance(movie, dict):
            continue

        name = str(
            movie.get("movieName") or ""
        ).strip()

        poster = str(
            movie.get("poster") or ""
        ).strip()

        if not name:
            continue

        if not poster.startswith(
            ("http://", "https://")
        ):
            continue

        movies.append({
            "movieCode": str(movie_code),
            "movieName": name,
            "poster": poster,
            "slug": slugify(name),
            "releaseDate": movie.get(
                "releaseDate"
            ),
        })

    movies.sort(
        key=lambda x: (
            x["slug"],
            x["movieCode"],
        )
    )

    # --------------------------------------------------------
    # Duplicate slug handling
    # --------------------------------------------------------

    used = {}

    for movie in movies:

        slug = movie["slug"]

        if slug not in used:

            used[slug] = movie["movieCode"]
            continue

        # Same movie title / duplicate slug.
        movie["slug"] = (
            f"{slug}-{slugify(movie['movieCode'])}"
        )

    return movies


# ============================================================
# LOCAL FILE VALIDATION
# ============================================================

def is_valid_existing_file(
    path: Path,
) -> bool:

    try:

        if not path.exists():
            return False

        if not path.is_file():
            return False

        size = path.stat().st_size

        if size <= 0:
            return False

        if size > MAX_FILE_SIZE:
            return False

        # Make sure it is actually a valid JPEG.
        with Image.open(path) as image:

            image.verify()

        return True

    except Exception:

        return False


# ============================================================
# SHOULD DOWNLOAD?
# ============================================================

def should_download(
    movie: dict[str, Any],
    manifest: dict[str, Any],
) -> tuple[bool, str]:

    output_path = (
        OUTPUT_DIR /
        f"{movie['slug']}.jpg"
    )

    movie_code = movie["movieCode"]

    previous = manifest.get(
        movie_code
    )

    # --------------------------------------------------------
    # Fastest possible check:
    #
    # Existing valid file + same URL
    # = SKIP WITHOUT ANY IMAGE REQUEST
    # --------------------------------------------------------

    if (
        previous
        and previous.get("poster") == movie["poster"]
        and previous.get("slug") == movie["slug"]
        and output_path.exists()
    ):

        if is_valid_existing_file(
            output_path
        ):
            return False, "cached"

    # --------------------------------------------------------
    # Even without manifest, a valid existing file is useful.
    #
    # This is important when upgrading from the old script.
    # --------------------------------------------------------

    if (
        output_path.exists()
        and is_valid_existing_file(output_path)
        and not previous
    ):

        return False, "existing-file"

    # --------------------------------------------------------
    # Missing / changed / corrupt
    # --------------------------------------------------------

    if not output_path.exists():
        return True, "missing"

    if previous:
        if previous.get("poster") != movie["poster"]:
            return True, "url-changed"

        if previous.get("slug") != movie["slug"]:
            return True, "slug-changed"

    if output_path.stat().st_size > MAX_FILE_SIZE:
        return True, "too-large"

    return True, "invalid"


# ============================================================
# DOWNLOAD IMAGE
# ============================================================

def download_image(
    session: requests.Session,
    url: str,
) -> bytes:

    response = session.get(
        url,
        timeout=(
            CONNECT_TIMEOUT,
            READ_TIMEOUT,
        ),
    )

    response.raise_for_status()

    data = response.content

    if not data:
        raise ValueError(
            "Empty image response"
        )

    return data


# ============================================================
# OPEN IMAGE
# ============================================================

def open_image(
    raw: bytes,
) -> Image.Image:

    if len(raw) > 15 * 1024 * 1024:
        raise ValueError(
            "Source image larger than 15 MB"
        )

    image = Image.open(
        io.BytesIO(raw)
    )

    image.load()

    if (
        image.width *
        image.height
    ) > MAX_PIXELS:

        raise ValueError(
            "Image exceeds pixel limit"
        )

    image = ImageOps.exif_transpose(
        image
    )

    if image.mode != "RGB":

        if "A" in image.getbands():

            background = Image.new(
                "RGB",
                image.size,
                "white",
            )

            background.paste(
                image,
                mask=image.getchannel("A"),
            )

            image = background

        else:

            image = image.convert(
                "RGB"
            )

    return image


# ============================================================
# JPEG
# ============================================================

def encode_jpeg(
    image: Image.Image,
    quality: int,
) -> bytes:

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=quality,
        optimize=True,
        progressive=True,
        subsampling="4:2:0",
    )

    return buffer.getvalue()


def best_quality(
    image: Image.Image,
) -> tuple[bytes, int]:

    # Maximum quality first.
    encoded = encode_jpeg(
        image,
        MAX_QUALITY,
    )

    if len(encoded) <= MAX_FILE_SIZE:
        return encoded, MAX_QUALITY

    # Minimum quality.
    encoded = encode_jpeg(
        image,
        MIN_QUALITY,
    )

    if len(encoded) > MAX_FILE_SIZE:
        return encoded, MIN_QUALITY

    low = MIN_QUALITY
    high = MAX_QUALITY

    best_data = encoded
    best_quality = MIN_QUALITY

    while low <= high:

        quality = (
            low + high
        ) // 2

        encoded = encode_jpeg(
            image,
            quality,
        )

        if len(encoded) <= MAX_FILE_SIZE:

            best_data = encoded
            best_quality = quality

            low = quality + 1

        else:

            high = quality - 1

    return (
        best_data,
        best_quality,
    )


# ============================================================
# COMPRESS
# ============================================================

def compress_image(
    image: Image.Image,
) -> tuple[
    bytes,
    int,
    int,
    int,
]:

    current = image

    while True:

        encoded, quality = best_quality(
            current
        )

        if len(encoded) <= MAX_FILE_SIZE:

            return (
                encoded,
                quality,
                current.width,
                current.height,
            )

        new_width = int(
            current.width *
            RESIZE_FACTOR
        )

        new_height = int(
            current.height *
            RESIZE_FACTOR
        )

        if (
            new_width < MIN_WIDTH
            or new_height < MIN_HEIGHT
        ):

            raise ValueError(
                "Cannot reach 20 KB limit"
            )

        current = current.resize(
            (
                new_width,
                new_height,
            ),
            Image.Resampling.LANCZOS,
        )


# ============================================================
# ATOMIC WRITE
# ============================================================

def atomic_write(
    path: Path,
    data: bytes,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=".tmp",
        dir=path.parent,
    )

    try:

        with os.fdopen(
            fd,
            "wb",
        ) as f:

            f.write(data)
            f.flush()
            os.fsync(f.fileno())

        os.replace(
            temp_name,
            path,
        )

    finally:

        try:
            os.unlink(
                temp_name
            )
        except FileNotFoundError:
            pass


# ============================================================
# PROCESS ONE MOVIE
# ============================================================

def process_movie(
    movie: dict[str, Any],
) -> dict[str, Any]:

    output_path = (
        OUTPUT_DIR /
        f"{movie['slug']}.jpg"
    )

    session = create_session()

    try:

        raw = download_image(
            session,
            movie["poster"],
        )

        source_hash = hashlib.sha256(
            raw
        ).hexdigest()

        image = open_image(
            raw
        )

        original_width = image.width
        original_height = image.height

        (
            encoded,
            quality,
            width,
            height,
        ) = compress_image(
            image
        )

        if len(encoded) > MAX_FILE_SIZE:
            raise ValueError(
                f"Generated image is "
                f"{len(encoded)} bytes"
            )

        atomic_write(
            output_path,
            encoded,
        )

        return {
            "movieCode": movie["movieCode"],
            "movieName": movie["movieName"],
            "slug": movie["slug"],
            "poster": movie["poster"],
            "file": f"/images/{movie['slug']}.jpg",
            "size": len(encoded),
            "sizeKB": round(
                len(encoded) / 1024,
                2,
            ),
            "quality": quality,
            "width": width,
            "height": height,
            "originalWidth": original_width,
            "originalHeight": original_height,
            "sourceHash": source_hash,
            "status": "downloaded",
        }

    except Exception as exc:

        return {
            "movieCode": movie["movieCode"],
            "movieName": movie["movieName"],
            "slug": movie["slug"],
            "poster": movie["poster"],
            "file": f"/images/{movie['slug']}.jpg",
            "status": "failed",
            "error": str(exc),
        }

    finally:

        session.close()


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    started = time.perf_counter()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load previous manifest
    # --------------------------------------------------------

    manifest = load_manifest()

    logger.info(
        "Previous manifest entries: %d",
        len(manifest),
    )

    # --------------------------------------------------------
    # Fetch database
    # --------------------------------------------------------

    session = create_session()

    try:

        database = download_database(
            session
        )

    except Exception as exc:

        logger.error(
            "Failed to fetch database: %s",
            exc,
        )

        return 1

    finally:

        session.close()

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    movies = normalize_movies(
        database
    )

    logger.info(
        "Movies in source: %d",
        len(movies),
    )

    # --------------------------------------------------------
    # Determine required work
    # --------------------------------------------------------

    to_download = []
    skipped = 0

    for movie in movies:

        needs_download, reason = should_download(
            movie,
            manifest,
        )

        if needs_download:

            movie["_reason"] = reason

            to_download.append(
                movie
            )

        else:

            skipped += 1

    logger.info(
        "Already cached : %d",
        skipped,
    )

    logger.info(
        "Need download  : %d",
        len(to_download),
    )

    # --------------------------------------------------------
    # Nothing to do
    # --------------------------------------------------------

    if not to_download:

        logger.info(
            "Everything is already up to date."
        )

        logger.info(
            "No poster requests will be made."
        )

        # Still clean/update manifest.
        save_manifest(
            {
                movie["movieCode"]: {
                    **manifest.get(
                        movie["movieCode"],
                        {},
                    ),
                    "movieCode": movie[
                        "movieCode"
                    ],
                    "movieName": movie[
                        "movieName"
                    ],
                    "slug": movie[
                        "slug"
                    ],
                    "poster": movie[
                        "poster"
                    ],
                    "file": (
                        f"/images/"
                        f"{movie['slug']}.jpg"
                    ),
                }
                for movie in movies
                if (
                    is_valid_existing_file(
                        OUTPUT_DIR /
                        f"{movie['slug']}.jpg"
                    )
                )
            }
        )

        elapsed = (
            time.perf_counter()
            - started
        )

        logger.info(
            "Completed in %.2f seconds",
            elapsed,
        )

        return 0

    # --------------------------------------------------------
    # Download only missing/changed images
    # --------------------------------------------------------

    logger.info(
        "Starting %d workers for %d images...",
        WORKERS,
        len(to_download),
    )

    downloaded = 0
    failed = 0

    with ThreadPoolExecutor(
        max_workers=WORKERS
    ) as executor:

        futures = {
            executor.submit(
                process_movie,
                movie,
            ): movie
            for movie in to_download
        }

        for future in as_completed(
            futures
        ):

            movie = futures[future]

            try:

                result = future.result()

            except Exception as exc:

                result = {
                    "movieCode": movie[
                        "movieCode"
                    ],
                    "movieName": movie[
                        "movieName"
                    ],
                    "slug": movie[
                        "slug"
                    ],
                    "poster": movie[
                        "poster"
                    ],
                    "status": "failed",
                    "error": str(exc),
                }

            if result["status"] == "downloaded":

                downloaded += 1

                manifest[
                    movie["movieCode"]
                ] = result

                logger.info(
                    "OK | %6.2f KB | Q%-2d | %s",
                    result["sizeKB"],
                    result["quality"],
                    movie["movieName"][:60],
                )

            else:

                failed += 1

                logger.error(
                    "FAILED | %s | %s",
                    movie["movieName"],
                    result.get(
                        "error",
                        "unknown",
                    ),
                )

    # --------------------------------------------------------
    # Make sure cached movies are also in manifest
    # --------------------------------------------------------

    for movie in movies:

        output_path = (
            OUTPUT_DIR /
            f"{movie['slug']}.jpg"
        )

        if is_valid_existing_file(
            output_path
        ):

            existing = manifest.get(
                movie["movieCode"],
                {},
            )

            manifest[
                movie["movieCode"]
            ] = {
                **existing,
                "movieCode": movie[
                    "movieCode"
                ],
                "movieName": movie[
                    "movieName"
                ],
                "slug": movie[
                    "slug"
                ],
                "poster": movie[
                    "poster"
                ],
                "file": (
                    f"/images/"
                    f"{movie['slug']}.jpg"
                ),
                "size": output_path.stat().st_size,
                "sizeKB": round(
                    output_path.stat().st_size
                    / 1024,
                    2,
                ),
            }

    # --------------------------------------------------------
    # Save manifest
    # --------------------------------------------------------

    save_manifest(
        manifest
    )

    elapsed = (
        time.perf_counter()
        - started
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    logger.info("")
    logger.info(
        "=========================================="
    )
    logger.info(
        " POSTER UPDATE COMPLETE"
    )
    logger.info(
        "=========================================="
    )

    logger.info(
        "Source movies  : %d",
        len(movies),
    )

    logger.info(
        "Cached/skipped  : %d",
        skipped,
    )

    logger.info(
        "Downloaded      : %d",
        downloaded,
    )

    logger.info(
        "Failed          : %d",
        failed,
    )

    logger.info(
        "Time            : %.2f sec",
        elapsed,
    )

    logger.info(
        "=========================================="
    )

    # Don't fail the whole workflow because
    # one or two BMS images temporarily failed.
    return 0


if __name__ == "__main__":
    sys.exit(main())
