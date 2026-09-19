#!/usr/bin/env python3

"""
BFILMY Production Movie Poster Pipeline
=======================================

SOURCE:
moviesdb.json

OUTPUT:
images/<movie-slug>.jpg

IMAGE:
400x600 portrait (40:60 / 2:3)

MAX SIZE:
20 KB

BEHAVIOUR:

1. Fetch moviesdb.json once.
2. Check local manifest/files.
3. Existing valid poster:
      -> SKIP
      -> NO BMS REQUEST

4. New movie:
      -> Download BMS poster

5. BMS poster download fails:
      -> Generate BFILMY fallback poster
      -> Save it
      -> Do NOT retry every run

6. Poster URL changes:
      -> Download new poster

7. Existing fallback:
      -> SKIP
      -> NO BMS REQUEST

8. GitHub Actions stages ALL files and pushes them.
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
from PIL import Image, ImageDraw, ImageFont, ImageOps
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

LOGO_URL = (
    "https://bfilmy.pages.dev/newlogo.png"
)

OUTPUT_DIR = Path("images")

MANIFEST_FILE = Path(
    "poster-manifest.json"
)

# ------------------------------------------------------------
# Image size
# 40:60 = 2:3
# ------------------------------------------------------------

POSTER_WIDTH = 400
POSTER_HEIGHT = 600

# ------------------------------------------------------------
# Maximum output size
# ------------------------------------------------------------

MAX_FILE_SIZE = 20 * 1024

# ------------------------------------------------------------
# Concurrent downloads
# ------------------------------------------------------------

WORKERS = 32

# ------------------------------------------------------------
# HTTP
# ------------------------------------------------------------

CONNECT_TIMEOUT = 10
READ_TIMEOUT = 30

MAX_RETRIES = 4
BACKOFF_FACTOR = 0.5

# ------------------------------------------------------------
# JPEG quality
# ------------------------------------------------------------

MIN_QUALITY = 25
MAX_QUALITY = 95

# ------------------------------------------------------------
# Resize
# ------------------------------------------------------------

RESIZE_FACTOR = 0.90

MIN_WIDTH = 120
MIN_HEIGHT = 180

MAX_PIXELS = 20_000_000

# ------------------------------------------------------------
# User agent
# ------------------------------------------------------------

USER_AGENT = (
    "BFILMY-Poster-Downloader/4.0 "
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

    value = unicodedata.normalize(
        "NFKD",
        value,
    )

    value = value.encode(
        "ascii",
        "ignore",
    ).decode(
        "ascii"
    )

    value = value.lower()

    value = value.replace(
        "&",
        " and ",
    )

    value = value.replace(
        "'",
        "",
    )

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

    return (
        value.strip("-")
        or "movie"
    )


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

        if not isinstance(
            data,
            dict,
        ):
            return {}

        return data

    except Exception as exc:

        logger.warning(
            "Manifest load failed: %s",
            exc,
        )

        return {}


def save_manifest(
    manifest: dict[str, Any],
) -> None:

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
# DATABASE
# ============================================================

def download_database(
    session: requests.Session,
) -> dict[str, Any]:

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
            "moviesdb.json returned empty response"
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
        len(raw) / (1024 * 1024),
    )

    try:

        text = raw.decode(
            "utf-8-sig"
        )

    except UnicodeDecodeError as exc:

        raise ValueError(
            "moviesdb.json is not UTF-8"
        ) from exc

    text = text.strip()

    if not text:

        raise ValueError(
            "moviesdb.json is empty"
        )

    try:

        data = json.loads(
            text
        )

    except json.JSONDecodeError as exc:

        logger.error(
            "JSON parsing failed."
        )

        logger.error(
            "Response preview:"
        )

        logger.error(
            "%s",
            text[:1000].replace(
                "\n",
                "\\n",
            ),
        )

        raise ValueError(
            "moviesdb.json is not valid JSON"
        ) from exc

    if not isinstance(
        data,
        dict,
    ):

        raise ValueError(
            "moviesdb.json root must be an object"
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

    movies = []

    for movie_code, movie in data.items():

        if not isinstance(
            movie,
            dict,
        ):
            continue

        name = str(
            movie.get(
                "movieName"
            ) or ""
        ).strip()

        poster = str(
            movie.get(
                "poster"
            ) or ""
        ).strip()

        if not name:
            continue

        if not poster.startswith(
            (
                "http://",
                "https://",
            )
        ):
            continue

        movies.append({
            "movieCode": str(
                movie_code
            ),

            "movieName": name,

            "poster": poster,

            "slug": slugify(
                name
            ),

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
    # Duplicate slug protection
    # --------------------------------------------------------

    used = {}

    for movie in movies:

        slug = movie[
            "slug"
        ]

        if slug not in used:

            used[
                slug
            ] = movie[
                "movieCode"
            ]

            continue

        movie[
            "slug"
        ] = (
            f"{slug}-"
            f"{slugify(movie['movieCode'])}"
        )

    return movies


# ============================================================
# VALIDATE LOCAL IMAGE
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

        with Image.open(
            path
        ) as image:

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
    # Existing valid file
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
    # Existing file with no manifest
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
    # Poster URL changed
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
    # Invalid file
    # --------------------------------------------------------

    return (
        True,
        "invalid",
    )


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
        allow_redirects=True,
    )

    response.raise_for_status()

    data = response.content

    if not data:

        raise ValueError(
            "Empty poster response"
        )

    return data


# ============================================================
# OPEN IMAGE
# ============================================================

def open_image(
    raw: bytes,
) -> Image.Image:

    if len(raw) > (
        15 * 1024 * 1024
    ):

        raise ValueError(
            "Source image exceeds 15 MB"
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
# JPEG ENCODING
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
# BEST QUALITY UNDER 20 KB
# ============================================================

def best_quality(
    image: Image.Image,
) -> tuple[
    bytes,
    int,
]:

    encoded = encode_jpeg(
        image,
        MAX_QUALITY,
    )

    if len(encoded) <= MAX_FILE_SIZE:

        return (
            encoded,
            MAX_QUALITY,
        )

    encoded = encode_jpeg(
        image,
        MIN_QUALITY,
    )

    if len(encoded) > MAX_FILE_SIZE:

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

            low = quality + 1

        else:

            high = quality - 1

    return (
        best_data,
        best_quality,
    )


# ============================================================
# COMPRESS REAL POSTER
# ============================================================

def compress_image(
    image: Image.Image,
) -> tuple[
    bytes,
    int,
    int,
    int,
]:

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # We first resize to a reasonable portrait size.
    #
    # 400x600 is enough for your website while giving
    # JPEG compression a much better chance to stay under
    # 20 KB.
    # --------------------------------------------------------

    current = image.copy()

    # Preserve aspect ratio.
    ratio = min(
        POSTER_WIDTH / current.width,
        POSTER_HEIGHT / current.height,
    )

    new_width = max(
        1,
        int(current.width * ratio),
    )

    new_height = max(
        1,
        int(current.height * ratio),
    )

    current = current.resize(
        (
            new_width,
            new_height,
        ),
        Image.Resampling.LANCZOS,
    )

    # Put into exact 400x600 canvas.
    canvas = Image.new(
        "RGB",
        (
            POSTER_WIDTH,
            POSTER_HEIGHT,
        ),
        "black",
    )

    x = (
        POSTER_WIDTH -
        current.width
    ) // 2

    y = (
        POSTER_HEIGHT -
        current.height
    ) // 2

    canvas.paste(
        current,
        (
            x,
            y,
        ),
    )

    current = canvas

    # --------------------------------------------------------
    # Find best quality.
    # --------------------------------------------------------

    while True:

        encoded, quality = (
            best_quality(
                current
            )
        )

        if len(encoded) <= MAX_FILE_SIZE:

            return (
                encoded,
                quality,
                current.width,
                current.height,
            )

        # ----------------------------------------------------
        # Still too large -> resize.
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
                "Unable to compress "
                "poster below 20 KB"
            )

        current = current.resize(
            (
                new_width,
                new_height,
            ),
            Image.Resampling.LANCZOS,
        )


# ============================================================
# FONT
# ============================================================

def get_font(
    size: int,
    bold: bool = False,
):

    candidates = []

    if bold:

        candidates.extend([
            "/usr/share/fonts/truetype/dejavu/"
            "DejaVuSans-Bold.ttf",

            "/usr/share/fonts/truetype/liberation2/"
            "LiberationSans-Bold.ttf",
        ])

    else:

        candidates.extend([
            "/usr/share/fonts/truetype/dejavu/"
            "DejaVuSans.ttf",

            "/usr/share/fonts/truetype/liberation2/"
            "LiberationSans-Regular.ttf",
        ])

    for path in candidates:

        if Path(path).exists():

            return ImageFont.truetype(
                path,
                size,
            )

    return ImageFont.load_default()


# ============================================================
# TEXT WRAPPING
# ============================================================

def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font,
    max_width: int,
) -> list[str]:

    words = text.split()

    if not words:
        return ["Movie"]

    lines = []
    current = ""

    for word in words:

        test = (
            word
            if not current
            else f"{current} {word}"
        )

        bbox = draw.textbbox(
            (0, 0),
            test,
            font=font,
        )

        width = (
            bbox[2] -
            bbox[0]
        )

        if width <= max_width:

            current = test

        else:

            if current:
                lines.append(
                    current
                )

            current = word

    if current:
        lines.append(
            current
        )

    return lines


# ============================================================
# DOWNLOAD LOGO
# ============================================================

def download_logo(
    session: requests.Session,
) -> Image.Image | None:

    try:

        logger.info(
            "Downloading BFILMY logo..."
        )

        response = session.get(
            LOGO_URL,
            timeout=(
                CONNECT_TIMEOUT,
                READ_TIMEOUT,
            ),
        )

        response.raise_for_status()

        image = Image.open(
            io.BytesIO(
                response.content
            )
        )

        image.load()

        image = ImageOps.exif_transpose(
            image
        )

        if image.mode != "RGBA":

            image = image.convert(
                "RGBA"
            )

        return image

    except Exception as exc:

        logger.warning(
            "Logo download failed: %s",
            exc,
        )

        return None


# ============================================================
# FALLBACK POSTER
# ============================================================

def create_fallback_poster(
    movie_name: str,
    logo: Image.Image | None,
) -> tuple[
    bytes,
    int,
    int,
    int,
]:

    width = POSTER_WIDTH
    height = POSTER_HEIGHT

    image = Image.new(
        "RGB",
        (
            width,
            height,
        ),
        "#090909",
    )

    draw = ImageDraw.Draw(
        image
    )

    # --------------------------------------------------------
    # Background
    # --------------------------------------------------------

    # Subtle vertical gradient.
    for y in range(height):

        value = int(
            8 +
            (
                22 *
                y /
                height
            )
        )

        draw.line(
            [
                (0, y),
                (width, y),
            ],
            fill=(
                value,
                value,
                value + 8,
            ),
        )

    # --------------------------------------------------------
    # Accent border
    # --------------------------------------------------------

    draw.rounded_rectangle(
        (
            12,
            12,
            width - 12,
            height - 12,
        ),
        radius=18,
        outline="#D92BFF",
        width=3,
    )

    # --------------------------------------------------------
    # Logo
    # --------------------------------------------------------

    if logo is not None:

        logo_copy = logo.copy()

        max_logo_width = 230
        max_logo_height = 170

        ratio = min(
            max_logo_width /
            logo_copy.width,

            max_logo_height /
            logo_copy.height,
        )

        logo_width = max(
            1,
            int(
                logo_copy.width *
                ratio
            ),
        )

        logo_height = max(
            1,
            int(
                logo_copy.height *
                ratio
            ),
        )

        logo_copy = logo_copy.resize(
            (
                logo_width,
                logo_height,
            ),
            Image.Resampling.LANCZOS,
        )

        logo_x = (
            width -
            logo_width
        ) // 2

        logo_y = 45

        image_rgba = image.convert(
            "RGBA"
        )

        image_rgba.alpha_composite(
            logo_copy,
            (
                logo_x,
                logo_y,
            ),
        )

        image = image_rgba.convert(
            "RGB"
        )

        draw = ImageDraw.Draw(
            image
        )

    # --------------------------------------------------------
    # Movie name
    # --------------------------------------------------------

    name_font_size = 42

    if len(movie_name) > 35:
        name_font_size = 34

    if len(movie_name) > 55:
        name_font_size = 28

    if len(movie_name) > 75:
        name_font_size = 24

    name_font = get_font(
        name_font_size,
        bold=True,
    )

    lines = wrap_text(
        draw,
        movie_name,
        name_font,
        330,
    )

    # Limit lines.
    lines = lines[:5]

    line_height = (
        name_font_size + 10
    )

    total_height = (
        len(lines) *
        line_height
    )

    start_y = 300 - (
        total_height // 2
    )

    for index, line in enumerate(
        lines
    ):

        bbox = draw.textbbox(
            (0, 0),
            line,
            font=name_font,
        )

        text_width = (
            bbox[2] -
            bbox[0]
        )

        x = (
            width -
            text_width
        ) // 2

        y = (
            start_y +
            index *
            line_height
        )

        # Shadow.
        draw.text(
            (
                x + 2,
                y + 2,
            ),
            line,
            font=name_font,
            fill="#000000",
        )

        # Text.
        draw.text(
            (
                x,
                y,
            ),
            line,
            font=name_font,
            fill="#FFFFFF",
        )

    # --------------------------------------------------------
    # Unavailable label
    # --------------------------------------------------------

    label_font = get_font(
        19,
        bold=True,
    )

    label = "POSTER UNAVAILABLE"

    bbox = draw.textbbox(
        (0, 0),
        label,
        font=label_font,
    )

    label_width = (
        bbox[2] -
        bbox[0]
    )

    label_x = (
        width -
        label_width
    ) // 2

    label_y = 470

    draw.rounded_rectangle(
        (
            label_x - 18,
            label_y - 10,
            label_x +
            label_width +
            18,
            label_y + 34,
        ),
        radius=10,
        fill="#161616",
        outline="#555555",
        width=1,
    )

    draw.text(
        (
            label_x,
            label_y,
        ),
        label,
        font=label_font,
        fill="#CCCCCC",
    )

    # --------------------------------------------------------
    # BFILMY branding
    # --------------------------------------------------------

    brand_font = get_font(
        17,
        bold=True,
    )

    brand = "BFILMY"

    bbox = draw.textbbox(
        (0, 0),
        brand,
        font=brand_font,
    )

    brand_width = (
        bbox[2] -
        bbox[0]
    )

    draw.text(
        (
            (
                width -
                brand_width
            ) // 2,
            545,
        ),
        brand,
        font=brand_font,
        fill="#FFFFFF",
    )

    # --------------------------------------------------------
    # Compress fallback
    # --------------------------------------------------------

    return compress_image(
        image
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
# PROCESS MOVIE
# ============================================================

def process_movie(
    movie: dict[str, Any],
    logo: Image.Image | None,
) -> dict[str, Any]:

    output_path = (
        OUTPUT_DIR /
        f"{movie['slug']}.jpg"
    )

    session = create_session()

    started = time.perf_counter()

    try:

        # ----------------------------------------------------
        # Try real BMS poster
        # ----------------------------------------------------

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
                    "Generated poster exceeds 20 KB"
                )

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

                "sourceHash": source_hash,

                "type": "original",

                "status": "downloaded",

                "seconds": round(
                    elapsed,
                    3,
                ),
            }

        except Exception as poster_error:

            # ------------------------------------------------
            # BMS poster failed.
            #
            # Generate local fallback.
            # ------------------------------------------------

            logger.warning(
                "BMS poster failed | %s | %s",
                movie[
                    "movieName"
                ],
                poster_error,
            )

            (
                fallback,
                quality,
                width,
                height,
            ) = create_fallback_poster(
                movie[
                    "movieName"
                ],
                logo,
            )

            if len(fallback) > MAX_FILE_SIZE:

                raise ValueError(
                    "Fallback poster exceeds 20 KB"
                )

            atomic_write(
                output_path,
                fallback,
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

                "size": len(fallback),

                "sizeKB": round(
                    len(fallback) / 1024,
                    2,
                ),

                "quality": quality,

                "width": width,

                "height": height,

                "type": "fallback",

                "fallbackReason": str(
                    poster_error
                ),

                "status": "fallback",

                "seconds": round(
                    elapsed,
                    3,
                ),
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
    # Database
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
    # Download logo ONCE
    # --------------------------------------------------------

    session = create_session()

    try:

        logo = download_logo(
            session
        )

    finally:

        session.close()

    if logo is None:

        logger.warning(
            "BFILMY logo unavailable. "
            "Fallback posters will use text branding."
        )

    # --------------------------------------------------------
    # Determine delta
    # --------------------------------------------------------

    to_process = []

    cached = 0

    reasons = {}

    for movie in movies:

        needs_download, reason = (
            should_download(
                movie,
                manifest,
            )
        )

        if needs_download:

            movie[
                "_reason"
            ] = reason

            to_process.append(
                movie
            )

            reasons[
                reason
            ] = (
                reasons.get(
                    reason,
                    0,
                ) + 1
            )

        else:

            cached += 1

    logger.info(
        "Already cached : %d",
        cached,
    )

    logger.info(
        "Need process   : %d",
        len(to_process),
    )

    if reasons:

        for reason, count in sorted(
            reasons.items()
        ):

            logger.info(
                "  %-15s %d",
                reason,
                count,
            )

    # --------------------------------------------------------
    # Nothing to do
    # --------------------------------------------------------

    if not to_process:

        logger.info(
            "Everything is already up to date."
        )

        logger.info(
            "ZERO poster requests."
        )

        # Refresh manifest metadata.
        for movie in movies:

            output_path = (
                OUTPUT_DIR /
                f"{movie['slug']}.jpg"
            )

            if not is_valid_existing_file(
                output_path
            ):
                continue

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

        logger.info(
            "Completed in %.2f seconds.",
            time.perf_counter()
            - started,
        )

        return 0

    # --------------------------------------------------------
    # Process only delta
    # --------------------------------------------------------

    logger.info(
        "Starting %d workers...",
        WORKERS,
    )

    original_count = 0
    fallback_count = 0

    with ThreadPoolExecutor(
        max_workers=WORKERS
    ) as executor:

        futures = {
            executor.submit(
                process_movie,
                movie,
                logo,
            ): movie
            for movie in to_process
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

                logger.error(
                    "Worker failed | %s | %s",
                    movie[
                        "movieName"
                    ],
                    exc,
                )

                continue

            # ------------------------------------------------
            # ORIGINAL
            # ------------------------------------------------

            if result[
                "type"
            ] == "original":

                original_count += 1

                logger.info(
                    "ORIGINAL | "
                    "%6.2f KB | "
                    "Q%-2d | "
                    "%s",

                    result[
                        "sizeKB"
                    ],

                    result[
                        "quality"
                    ],

                    movie[
                        "movieName"
                    ][:60],
                )

            # ------------------------------------------------
            # FALLBACK
            # ------------------------------------------------

            else:

                fallback_count += 1

                logger.warning(
                    "FALLBACK | "
                    "%6.2f KB | "
                    "%s",

                    result[
                        "sizeKB"
                    ],

                    movie[
                        "movieName"
                    ][:60],
                )

            # ------------------------------------------------
            # Save result
            # ------------------------------------------------

            manifest[
                movie[
                    "movieCode"
                ]
            ] = result

    # --------------------------------------------------------
    # Refresh manifest
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

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_manifest(
        manifest
    )

    elapsed = (
        time.perf_counter()
        - started
    )

    logger.info("")
    logger.info(
        "=========================================="
    )

    logger.info(
        "BFILMY POSTER PIPELINE COMPLETE"
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
        "Original       : %d",
        original_count,
    )

    logger.info(
        "Fallback       : %d",
        fallback_count,
    )

    logger.info(
        "Total time     : %.2f sec",
        elapsed,
    )

    logger.info(
        "=========================================="
    )

    return 0


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    sys.exit(
        main()
    )
