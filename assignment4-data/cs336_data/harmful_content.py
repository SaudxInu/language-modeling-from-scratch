import sys

import fasttext
from fastwarc.warc import ArchiveIterator, WarcRecordType

from cs336_data.extract_text import extract_text


def detect_nsfw(text: str) -> tuple[str, float]:
    model = fasttext.FastText.load_model("data/jigsaw_fasttext_bigrams_nsfw_final.bin")
    response = model.predict(text.replace("\n", " "))
    return response[0][0].replace("__label__", ""), response[1][0]


def detect_hate_speech(text: str) -> tuple[str, float]:
    model = fasttext.FastText.load_model(
        "data/jigsaw_fasttext_bigrams_hatespeech_final.bin"
    )
    response = model.predict(text.replace("\n", " "))
    return response[0][0].replace("__label__", ""), response[1][0]


def main(warc_file_path: str):
    with open(warc_file_path, "rb") as warc_file:
        k = 0
        for record in ArchiveIterator(warc_file):
            if record.record_type == WarcRecordType.response:
                html_bytes = record.reader.read()
                text = extract_text(html_bytes)
                text = text.replace("\n", " ")
                is_nsfw, score_nsfw = detect_nsfw(text)
                is_hate_speech, score_hate_speech = detect_hate_speech(text)
                if is_nsfw == "nsfw" or is_hate_speech == "toxic":
                    record = text[:100]
                    print("=" * 20)
                    print(
                        f"Record {record}: NSFW: {is_nsfw} (Score: {score_nsfw}), "
                        f"Hate Speech: {is_hate_speech} (Score: {score_hate_speech})"
                    )
                    k += 1
            if k >= 5:
                break


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python harmful_content.py <warc_file_path>")
        sys.exit(1)
    warc_file_path = sys.argv[1]
    main(warc_file_path)
