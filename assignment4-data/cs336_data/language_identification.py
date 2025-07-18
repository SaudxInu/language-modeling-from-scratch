import sys

import fasttext
from fastwarc.warc import ArchiveIterator, WarcRecordType

from cs336_data.extract_text import extract_text


def language_identification(text: str) -> tuple[str, float]:
    model = fasttext.FastText.load_model("data/lid.176.bin")
    response = model.predict(text.replace("\n", " "))
    return response[0][0].replace("__label__", ""), max(0.0, min(1.0, response[1][0]))


def main(warc_file_path: str):
    with open(warc_file_path, "rb") as warc_file:
        k = 0
        for record in ArchiveIterator(warc_file):
            if record.record_type == WarcRecordType.response:
                html_bytes = record.reader.read()
                text = extract_text(html_bytes)
                language, confidence = language_identification(text)
                record = text.replace("\n", " ")[:100]
                print("=" * 20)
                print(
                    f"Record {record}: Language: {language}, Confidence: {confidence}"
                )
                k += 1
                if k >= 20:
                    break


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python language_identification.py <warc_file_path>")
        sys.exit(1)
    warc_file_path = sys.argv[1]
    main(warc_file_path)
