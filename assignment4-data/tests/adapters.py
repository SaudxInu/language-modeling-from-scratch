from __future__ import annotations

import os
from typing import Any

from cs336_data.exact_deduplication import exact_deduplication
from cs336_data.gopher_quality_filters import gopher_quality_filters
from cs336_data.harmful_content import detect_hate_speech, detect_nsfw
from cs336_data.language_identification import language_identification
from cs336_data.extract_text import extract_text
from cs336_data.mask_pii import mask_email, mask_phone_number, mask_ip_address
from cs336_data.minhash_deduplication import minhash_deduplication
from cs336_data.quality_classifier import classify_quality


def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    return extract_text(html_bytes)


def run_identify_language(text: str) -> tuple[Any, float]:
    return language_identification(text)


def run_mask_emails(text: str) -> tuple[str, int]:
    return mask_email(text)


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    return mask_phone_number(text)


def run_mask_ips(text: str) -> tuple[str, int]:
    return mask_ip_address(text)


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    return detect_nsfw(text)


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    return detect_hate_speech(text)


def run_classify_quality(text: str) -> tuple[Any, float]:
    return classify_quality(text)


def run_gopher_quality_filter(text: str) -> bool:
    return gopher_quality_filters(text)


def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    return exact_deduplication(input_files, output_directory)


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    return minhash_deduplication(
        input_files, num_hashes, num_bands, ngrams, jaccard_threshold, output_directory
    )
