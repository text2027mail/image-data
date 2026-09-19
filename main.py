#!/usr/bin/env python3

"""
BFILMY Movie Poster Downloader
==============================

Source:
https://raw.githubusercontent.com/unknownman2024/bms-interest-track/main/Bookmyshow%20Data/moviesdb.json

Output:
images/<movie-slug>.jpg

Production features:
- 2,000+ movie support
- Incremental downloads
- Existing posters are NOT fetched again
- Only new/changed/missing posters are downloaded
- Persistent manifest
- 20 KB hard file-size limit
- Highest possible JPEG quality under 20 KB
- Progressive resizing only when necessary
- 32 concurrent downloads
- Connection pooling
- Automatic retries
- Exponential backoff
- Atomic file writes
- Corrupt-file detection
- Duplicate slug handling
- UTF-8 BOM support
- CDN/HTML error diagnostics
- GitHub Actions friendly
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
    "https://raw.githubusercontent.com/"
    "unknownman2024/"
    "bms-interest-track/"
    "main/"
    "Bookmyshow%20Data/"
    "moviesdb.json"
)

OUTPUT_DIR = Path("images")

MANIFEST_FILE = Path(
    "poster-manifest.json"
)

# Hard maximum image size.
MAX_FILE_SIZE = 20 * 1024

# Number of simultaneous image downloads.
WORKERS = 32

# HTTP timeout.
CONNECT_TIMEOUT = 10
READ_TIMEOUT = 30

# Retry settings.
MAX_RETRIES = 4
BACKOFF_FACTOR = 0.5

# JPEG quality range.
MIN_QUALITY = 25
MAX_QUALITY = 95

# If quality 25 still cannot fit under 20 KB,
# reduce dimensions by this percentage.
RESIZE_FACTOR = 0.90

# Never resize below these dimensions.
MIN_WIDTH = 120
MIN_HEIGHT = 180

# Protection against huge images.
MAX_PIXELS = 20_000_000

USER_AGENT = (
    "BFILMY-Poster-Downloader/3.0 "
    "(production)"
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(
    "bfilmy-posters"
)


# ============================================================
# HTTP SESSION
# ============================================================

def create_session() -> requests.Session:
    """
    Creates a requests session with:

    - Connection pooling
    - Automatic retries
    - Backoff
    - 429/5xx handling
    """

    session = requests.Session()

    retry = Retry(
        total=MAX_RETRIES,
        connect=MAX_RETRIES,
        read=MAX_RETRIES,
        status=MAX_RETRIES,

        backoff_factor=BACKOFF_FACTOR,

        status_forcelist=(
            429,
            500,
            502,
            503,
            504,
        ),

        allowed_methods=frozenset(
            ["GET"]
        ),

        respect_retry_after_header=True,

        raise_on_status=False,
    )

    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=WORKERS,
        pool_maxsize=WORKERS,
    )

    session.mount(
        "https://",
        adapter,
    )

    session.mount(
        "http://",
        adapter,
    )

    session.headers.update({
        "User-Agent": USER_AGENT,

        "Accept": (
            "application/json,"
            "image/avif,image/webp,"
            "image/apng,image/*,"
            "*/*;q=0.8"
        ),

        "Accept-Encoding": (
            "gzip, deflate, br"
        ),

        "Connection": "keep-alive",
    })

    return session


# ============================================================
# SLUGIFY
# ============================================================

def slugify(
    value: str,
) -> str:

    value = str(
        value or ""
    ).strip()

    # Normalize Unicode.
    value = unicodedata.normalize(
        "NFKD",
        value,
    )

    # Remove accents.
    value = value.encode(
        "ascii",
        "ignore",
    ).decode(
        "ascii"
    )

    value = value.lower()

    # & -> and
    value = value.replace(
        "&",
        " and ",
    )

    # Don't create extra hyphens for apostrophes.
    value = value.replace(
        "'",
        "",
    )

    # Everything else -> hyphen.
    value = re.sub(
        r"[^a-z0-9]+",
        "-",
        value,
    )

    # Remove duplicate hyphens.
    value = re.sub(
        r"-+",
        "-",
        value,
    )

    return value.strip(
        "-"
    ) or "movie"


# ============================================================
# MANIFEST
# ============================================================

def load_manifest() -> dict[str, Any]:
    """
    Loads the local poster manifest.

    If it doesn't exist, this is the first run.
    """

    if not MANIFEST_FILE.exists():

        return {}

    try:

        with MANIFEST_FILE.open(
            "r",
            encoding="utf-8",
        ) as f:

            data = json.load(f)

        if not isinstance(
            data,
            dict,
        ):

            logger.warning(
                "Manifest root isn't an object."
            )

            return {}

        return data

    except Exception as exc:

        logger.warning(
            "Could not load manifest: %s",
            exc,
        )

        return {}


def save_manifest(
    manifest: dict[str, Any],
) -> None:
    """
    Atomic manifest write.
    """

    MANIFEST_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = MANIFEST_FILE.with_suffix(
        ".tmp"
    )

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
# DOWNLOAD DATABASE
# ============================================================

def download_database(
    session: requests.Session,
) -> dict[str, Any]:
    """
    Downloads and parses moviesdb.json.

    Does NOT use response.json() directly.

    This handles:
    - UTF-8 BOM
    - CDN/proxy responses
    - Incorrect Content-Type
    - HTML error responses
    - Empty responses
    """

    logger.info(
        "Fetching moviesdb.json..."
    )

    response = session.get(
        SOURCE_URL,
        timeout=(
            CONNECT_TIMEOUT,
            READ_TIMEOUT,
        ),
        allow_redirects=True,
    )

    response.raise_for_status()

    raw = response.content

    if not raw:

        raise ValueError(
            "Source returned an empty response"
        )

    logger.info(
        "Source HTTP status : %s",
        response.status_code,
    )

    logger.info(
        "Source content-type: %s",
        response.headers.get(
            "content-type",
            "unknown",
        ),
    )

    logger.info(
        "Source size        : %.2f MB",
        len(raw) / (
            1024 * 1024
        ),
    )

    logger.info(
        "Source final URL   : %s",
        response.url,
    )

    # --------------------------------------------------------
    # Decode UTF-8.
    #
    # utf-8-sig automatically removes a BOM if present.
    # --------------------------------------------------------

    try:

        text = raw.decode(
            "utf-8-sig"
        )

    except UnicodeDecodeError as exc:

        raise ValueError(
            f"Source is not valid UTF-8: {exc}"
        ) from exc

    text = text.strip()

    if not text:

        raise ValueError(
            "Source returned empty text"
        )

    # --------------------------------------------------------
    # JSON parse
    # --------------------------------------------------------

    try:

        data = json.loads(
            text
        )

    except json.JSONDecodeError as exc:

        preview = (
            text[:1000]
            .replace(
                "\n",
                "\\n",
            )
        )

        logger.error(
            "JSON parsing failed."
        )

        logger.error(
            "Response preview:"
        )

        logger.error(
            "%s",
            preview,
        )

        raise ValueError(
            "moviesdb.json response "
            "is not valid JSON"
        ) from exc

    # --------------------------------------------------------
    # Validate root
    # --------------------------------------------------------

    if not isinstance(
        data,
        dict,
    ):

        raise ValueError(
            "moviesdb.json root "
            "must be an object"
        )

    logger.info(
        "JSON parsed successfully: %d movies",
        len(data),
    )

    return data


# ============================================================
# NORMALIZE MOVIES
# ============================================================

def normalize_movies(
    data: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Converts:

        {
            "ET00002396": {...}
        }

    into a normalized list.
    """

    movies: list[
        dict[str, Any]
    ] = []

    for movie_code, movie in data.items():

        if not isinstance(
            movie,
            dict,
        ):

            continue

        movie_name = str(
            movie.get(
                "movieName"
            ) or ""
        ).strip()

        poster = str(
            movie.get(
                "poster"
            ) or ""
        ).strip()

        if not movie_name:

            logger.warning(
                "Skipping %s: "
                "missing movieName",
                movie_code,
            )

            continue

        if not poster.startswith(
            (
                "http://",
                "https://",
            )
        ):

            logger.warning(
                "Skipping %s (%s): "
                "invalid poster URL",
                movie_code,
                movie_name,
            )

            continue

        movies.append({
            "movieCode": str(
                movie_code
            ),

            "movieName": movie_name,

            "poster": poster,

            "slug": slugify(
                movie_name
            ),

            "releaseDate": movie.get(
                "releaseDate"
            ),
        })

    # Deterministic order.
    movies.sort(
        key=lambda x: (
            x["slug"],
            x["movieCode"],
        )
    )

    # --------------------------------------------------------
    # Duplicate slug handling
    # --------------------------------------------------------

    used_slugs: dict[
        str,
        str
    ] = {}

    for movie in movies:

        slug = movie["slug"]

        if slug not in used_slugs:

            used_slugs[
                slug
            ] = movie[
                "movieCode"
            ]

            continue

        # Example:
        #
        # movie-name.jpg
        # movie-name-et123456.jpg

        movie["slug"] = (
            f"{slug}-"
            f"{slugify(movie['movieCode'])}"
        )

    return movies


# ============================================================
# EXISTING FILE VALIDATION
# ============================================================

def is_valid_existing_file(
    path: Path,
) -> bool:
    """
    Checks an existing poster without
    making a network request.
    """

    try:

        if not path.exists():
            return False

        if not path.is_file():
            return False

        size = path.stat().st_size

        # Empty or over 20 KB.
        if size <= 0:
            return False

        if size > MAX_FILE_SIZE:
            return False

        # Validate actual JPEG.
        with Image.open(path) as image:

            image.verify()

        return True

    except Exception:

        return False


# ============================================================
# CACHE DECISION
# ============================================================

def should_download(
    movie: dict[str, Any],
    manifest: dict[str, Any],
) -> tuple[
    bool,
    str,
]:
    """
    Determines whether this poster needs
    to be downloaded.

    IMPORTANT:

    A valid cached poster produces ZERO
    poster network requests.
    """

    movie_code = movie[
        "movieCode"
    ]

    slug = movie[
        "slug"
    ]

    output_path = (
        OUTPUT_DIR /
        f"{slug}.jpg"
    )

    previous = manifest.get(
        movie_code
    )

    # --------------------------------------------------------
    # BEST CASE
    #
    # Manifest says the local file was generated
    # from the same poster URL.
    # --------------------------------------------------------

    if (
        previous
        and previous.get(
            "poster"
        ) == movie["poster"]

        and previous.get(
            "slug"
        ) == slug

        and is_valid_existing_file(
            output_path
        )
    ):

        return (
            False,
            "cached",
        )

    # --------------------------------------------------------
    # Existing file but no manifest.
    #
    # Useful when upgrading from an older version.
    #
    # We don't unnecessarily download it.
    # --------------------------------------------------------

    if (
        not previous
        and is_valid_existing_file(
            output_path
        )
    ):

        return (
            False,
            "existing-file",
        )

    # --------------------------------------------------------
    # Missing
    # --------------------------------------------------------

    if not output_path.exists():

        return (
            True,
            "missing",
        )

    # --------------------------------------------------------
    # URL changed
    # --------------------------------------------------------

    if previous:

        if previous.get(
            "poster"
        ) != movie["poster"]:

            return (
                True,
                "url-changed",
            )

        if previous.get(
            "slug"
        ) != slug:

            return (
                True,
                "slug-changed",
            )

    # --------------------------------------------------------
    # Invalid / oversized
    # --------------------------------------------------------

    return (
        True,
        "invalid",
    )


# ============================================================
# DOWNLOAD POSTER
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
        allow_redirects=True,
    )

    response.raise_for_status()

    data = response.content

    if not data:

        raise ValueError(
            "Poster returned empty response"
        )

    return data


# ============================================================
# OPEN IMAGE
# ============================================================

def open_image(
    raw: bytes,
) -> Image.Image:

    # Prevent enormous source responses.
    if len(raw) > (
        15 * 1024 * 1024
    ):

        raise ValueError(
            "Source image exceeds 15 MB"
        )

    image = Image.open(
        io.BytesIO(raw)
    )

    # Force decode.
    image.load()

    if (
        image.width *
        image.height
    ) > MAX_PIXELS:

        raise ValueError(
            "Image exceeds maximum "
            "pixel count"
        )

    # Respect EXIF orientation.
    image = ImageOps.exif_transpose(
        image
    )

    # Convert to RGB.
    if image.mode != "RGB":

        if "A" in image.getbands():

            background = Image.new(
                "RGB",
                image.size,
                "white",
            )

            background.paste(
                image,
                mask=image.getchannel(
                    "A"
                ),
            )

            image = background

        else:

            image = image.convert(
                "RGB"
            )

    return image


# ============================================================
# JPEG ENCODE
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


# ============================================================
# HIGHEST QUALITY UNDER 20 KB
# ============================================================

def best_quality(
    image: Image.Image,
) -> tuple[
    bytes,
    int,
]:
    """
    Binary-searches JPEG quality.

    Returns the highest quality whose
    output is <= 20 KB.

    Much faster than trying every
    quality level.
    """

    # --------------------------------------------------------
    # Check maximum quality.
    # --------------------------------------------------------

    encoded = encode_jpeg(
        image,
        MAX_QUALITY,
    )

    if len(encoded) <= MAX_FILE_SIZE:

        return (
            encoded,
            MAX_QUALITY,
        )

    # --------------------------------------------------------
    # Check minimum quality.
    # --------------------------------------------------------

    encoded = encode_jpeg(
        image,
        MIN_QUALITY,
    )

    if len(encoded) > MAX_FILE_SIZE:

        # Even Q25 is too large.
        # Caller must resize.
        return (
            encoded,
            MIN_QUALITY,
        )

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

            # Try higher quality.
            low = quality + 1

        else:

            # Too large.
            high = quality - 1

    return (
        best_data,
        best_quality,
    )


# ============================================================
# COMPRESS TO <= 20 KB
# ============================================================

def compress_image(
    image: Image.Image,
) -> tuple[
    bytes,
    int,
    int,
    int,
]:
    """
    Attempts to preserve original dimensions.

    Only resizes when Q25 cannot reach
    the 20 KB limit.
    """

    current = image

    while True:

        encoded, quality = (
            best_quality(
                current
            )
        )

        # Success.
        if len(encoded) <= MAX_FILE_SIZE:

            return (
                encoded,
                quality,
                current.width,
                current.height,
            )

        # ----------------------------------------------------
        # Need smaller dimensions.
        # ----------------------------------------------------

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
                "Unable to produce "
                "poster under 20 KB"
            )

        current = current.resize(
            (
                new_width,
                new_height,
            ),
            Image.Resampling.LANCZOS,
        )


# ============================================================
# ATOMIC FILE WRITE
# ============================================================

def atomic_write(
    path: Path,
    data: bytes,
) -> None:
    """
    Prevents partially written images.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fd, temp_name = (
        tempfile.mkstemp(
            prefix=f".{path.stem}.",
            suffix=".tmp",
            dir=path.parent,
        )
    )

    try:

        with os.fdopen(
            fd,
            "wb",
        ) as f:

            f.write(data)

            f.flush()

            os.fsync(
                f.fileno()
            )

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
# PROCESS ONE POSTER
# ============================================================

def process_movie(
    movie: dict[str, Any],
) -> dict[str, Any]:

    output_path = (
        OUTPUT_DIR /
        f"{movie['slug']}.jpg"
    )

    session = create_session()

    started = time.perf_counter()

    try:

        # ----------------------------------------------------
        # Download
        # ----------------------------------------------------

        raw = download_image(
            session,
            movie["poster"],
        )

        source_hash = hashlib.sha256(
            raw
        ).hexdigest()

        # ----------------------------------------------------
        # Decode
        # ----------------------------------------------------

        image = open_image(
            raw
        )

        original_width = (
            image.width
        )

        original_height = (
            image.height
        )

        # ----------------------------------------------------
        # Compress
        # ----------------------------------------------------

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
                f"Generated image "
                f"is {len(encoded)} bytes"
            )

        # ----------------------------------------------------
        # Write
        # ----------------------------------------------------

        atomic_write(
            output_path,
            encoded,
        )

        elapsed = (
            time.perf_counter()
            - started
        )

        return {
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

            "size": len(encoded),

            "sizeKB": round(
                len(encoded) / 1024,
                2,
            ),

            "quality": quality,

            "width": width,

            "height": height,

            "originalWidth": (
                original_width
            ),

            "originalHeight": (
                original_height
            ),

            "sourceHash": source_hash,

            "status": "downloaded",

            "seconds": round(
                elapsed,
                3,
            ),
        }

    except Exception as exc:

        return {
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

    logger.info(
        "=========================================="
    )

    logger.info(
        "BFILMY POSTER PIPELINE"
    )

    logger.info(
        "=========================================="
    )

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load manifest
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
    # Normalize movies
    # --------------------------------------------------------

    movies = normalize_movies(
        database
    )

    if not movies:

        logger.error(
            "No valid movies found."
        )

        return 1

    logger.info(
        "Movies in source: %d",
        len(movies),
    )

    # --------------------------------------------------------
    # Determine required downloads
    # --------------------------------------------------------

    to_download: list[
        dict[str, Any]
    ] = []

    cached = 0

    reasons: dict[
        str,
        int
    ] = {}

    for movie in movies:

        needs_download, reason = (
            should_download(
                movie,
                manifest,
            )
        )

        if needs_download:

            movie["_reason"] = reason

            to_download.append(
                movie
            )

            reasons[reason] = (
                reasons.get(
                    reason,
                    0,
                ) + 1
            )

        else:

            cached += 1

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    logger.info(
        "Already cached : %d",
        cached,
    )

    logger.info(
        "Need download  : %d",
        len(to_download),
    )

    if reasons:

        logger.info(
            "Download reasons:"
        )

        for reason, count in sorted(
            reasons.items()
        ):

            logger.info(
                "  %-15s %d",
                reason,
                count,
            )

    # --------------------------------------------------------
    # Nothing changed
    # --------------------------------------------------------

    if not to_download:

        logger.info(
            "Everything is already up to date."
        )

        logger.info(
            "ZERO poster downloads required."
        )

        # Ensure all existing movies have
        # manifest entries.
        for movie in movies:

            output_path = (
                OUTPUT_DIR /
                f"{movie['slug']}.jpg"
            )

            if is_valid_existing_file(
                output_path
            ):

                previous = manifest.get(
                    movie[
                        "movieCode"
                    ],
                    {},
                )

                manifest[
                    movie[
                        "movieCode"
                    ]
                ] = {
                    **previous,

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

        save_manifest(
            manifest
        )

        elapsed = (
            time.perf_counter()
            - started
        )

        logger.info(
            "Completed in %.2f seconds.",
            elapsed,
        )

        return 0

    # --------------------------------------------------------
    # Download only delta
    # --------------------------------------------------------

    logger.info(
        "Starting %d workers...",
        WORKERS,
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

            movie = futures[
                future
            ]

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

                    "file": (
                        f"/images/"
                        f"{movie['slug']}.jpg"
                    ),

                    "status": "failed",

                    "error": str(exc),
                }

            # ------------------------------------------------
            # Successful
            # ------------------------------------------------

            if result[
                "status"
            ] == "downloaded":

                downloaded += 1

                manifest[
                    movie[
                        "movieCode"
                    ]
                ] = result

                logger.info(
                    "OK | "
                    "%6.2f KB | "
                    "Q%-2d | "
                    "%4dx%-4d | "
                    "%s",

                    result[
                        "sizeKB"
                    ],

                    result[
                        "quality"
                    ],

                    result[
                        "width"
                    ],

                    result[
                        "height"
                    ],

                    movie[
                        "movieName"
                    ][:60],
                )

            # ------------------------------------------------
            # Failed
            # ------------------------------------------------

            else:

                failed += 1

                logger.error(
                    "FAILED | %s | %s",

                    movie[
                        "movieName"
                    ],

                    result.get(
                        "error",
                        "unknown error",
                    ),
                )

    # --------------------------------------------------------
    # Refresh manifest for all valid local files
    # --------------------------------------------------------

    for movie in movies:

        output_path = (
            OUTPUT_DIR /
            f"{movie['slug']}.jpg"
        )

        if not is_valid_existing_file(
            output_path
        ):

            continue

        existing = manifest.get(
            movie[
                "movieCode"
            ],
            {},
        )

        manifest[
            movie[
                "movieCode"
            ]
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
    # Final summary
    # --------------------------------------------------------

    logger.info("")

    logger.info(
        "=========================================="
    )

    logger.info(
        "COMPLETE"
    )

    logger.info(
        "=========================================="
    )

    logger.info(
        "Source movies  : %d",
        len(movies),
    )

    logger.info(
        "Cached/skipped : %d",
        cached,
    )

    logger.info(
        "Downloaded     : %d",
        downloaded,
    )

    logger.info(
        "Failed         : %d",
        failed,
    )

    logger.info(
        "Total time     : %.2f sec",
        elapsed,
    )

    logger.info(
        "=========================================="
    )

    # --------------------------------------------------------
    # We don't fail the workflow because of a
    # temporary BMS poster failure.
    #
    # The failed movie will be retried on the
    # next workflow run.
    # --------------------------------------------------------

    return 0


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    sys.exit(
        main()
    )
