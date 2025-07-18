from fastwarc.warc import ArchiveIterator, WarcRecordType
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import pathlib

import pandas as pd

from cs336_data.minhash_deduplication import minhash_deduplication
from cs336_data.exact_deduplication import exact_deduplication
from cs336_data.extract_text import extract_text
from cs336_data.gopher_quality_filters import gopher_quality_filters
from cs336_data.harmful_content import detect_hate_speech, detect_nsfw
from cs336_data.language_identification import language_identification
from cs336_data.mask_pii import mask_email, mask_ip_address, mask_phone_number


def run_filtering(
    warc_input_file_path: str, base_output_file_path: str
) -> tuple[list[str], dict]:
    log = {
        "num_documents": 0,
        "extract_text": 0,
        "language_filtering": 0,
        "mask_pii": 0,
        "harmful_content": 0,
        "gopher_quality_filters": 0,
    }
    k = 0
    output_filepaths = []
    with open(warc_input_file_path, "rb") as warc_file:
        for record in tqdm(ArchiveIterator(warc_file), "Processing WARC file"):
            if record.record_type == WarcRecordType.response:
                log["num_documents"] += 1
                html_bytes = record.reader.read()
                text = extract_text(html_bytes)
                if not text:
                    log["extract_text"] += 1
                    continue
                text = text.replace("\n", " ")
                language, _ = language_identification(text)
                if language != "en":
                    log["language_filtering"] += 1
                    continue
                text, _ = mask_email(text)
                text, _ = mask_phone_number(text)
                text, _ = mask_ip_address(text)
                if not text:
                    log["mask_pii"] += 1
                    continue
                is_nsfw, _ = detect_nsfw(text)
                is_hate_speech, _ = detect_hate_speech(text)
                if is_nsfw == "nsfw" or is_hate_speech == "toxic":
                    log["harmful_content"] += 1
                    continue
                is_good_quality = gopher_quality_filters(text)
                if not is_good_quality:
                    log["gopher_quality_filters"] += 1
                    continue
                with open(base_output_file_path + f"_part-{k}.txt", "w") as f:
                    f.write(text + "\n")
                output_filepaths.append(base_output_file_path + f"_part-{k}.txt")
                k += 1
            if k >= 100:
                break
    return output_filepaths, log


def main():
    num_cpus = 1
    executor = ProcessPoolExecutor(max_workers=num_cpus)
    wet_filepaths = [
        "data/development.warc.gz",
    ]

    futures = []
    for wet_filepath in wet_filepaths:
        wet_filename = str(pathlib.Path(wet_filepath).name).split(".")[0]
        future = executor.submit(
            run_filtering, wet_filepath, f"data/filter/{wet_filename}"
        )
        futures.append(future)

    all_output_filepaths = []
    filtering_logs = []
    for future in tqdm(
        as_completed(futures),
        total=len(wet_filepaths),
        desc="Fetching results",
    ):
        output_filepaths, log = future.result()
        all_output_filepaths += output_filepaths
        filtering_logs.append(log)

    print("=" * 50)
    print("Filtering logs:")
    print(pd.DataFrame(filtering_logs).sum(axis=0))
    print("=" * 50)

    exact_deduplication_log = exact_deduplication(
        [pathlib.Path(f) for f in all_output_filepaths],
        pathlib.Path("data/exact_deduplication/"),
    )

    print("=" * 50)
    print(
        f"Documents after exact deduplication: {len(output_filepaths)}/{len(exact_deduplication_log)}"
    )
    print("=" * 50)

    output_filepaths = [
        pathlib.Path(os.path.join("data/exact_deduplication/", os.path.basename(fp)))
        for fp in exact_deduplication_log
    ]

    minhash_deduplication_log = minhash_deduplication(
        output_filepaths,
        100,
        10,
        2,
        0.5,
        pathlib.Path("data/minhash_deduplication/"),
    )

    print("=" * 50)
    print(
        f"Documents after minhash deduplication: {len(output_filepaths)}/{len(minhash_deduplication_log)}"
    )
    print("=" * 50)


if __name__ == "__main__":
    main()
