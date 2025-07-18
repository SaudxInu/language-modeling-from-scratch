import sys

import fasttext
from fastwarc.warc import ArchiveIterator, WarcRecordType

from cs336_data.extract_text import extract_text
from cs336_data.gopher_quality_filters import gopher_quality_filters
from cs336_data.harmful_content import detect_hate_speech, detect_nsfw
from cs336_data.language_identification import language_identification
from cs336_data.mask_pii import mask_email, mask_ip_address, mask_phone_number


def create_training_data(warc_file_path: str):
    with open("data/train_quality_classifier.txt", "w") as f:
        with open(warc_file_path, "rb") as warc_file:
            k = 0
            for record in ArchiveIterator(warc_file):
                if record.record_type == WarcRecordType.response:
                    html_bytes = record.reader.read()
                    text = extract_text(html_bytes)
                    text = text.replace("\n", " ")
                    language, _ = language_identification(text)
                    if language != "en":
                        continue
                    text, _ = mask_email(text)
                    text, _ = mask_phone_number(text)
                    text, _ = mask_ip_address(text)
                    is_nsfw, _ = detect_nsfw(text)
                    is_hate_speech, _ = detect_hate_speech(text)
                    if is_nsfw == "nsfw" or is_hate_speech == "toxic":
                        continue
                    is_good_quality = gopher_quality_filters(text)
                    if not is_good_quality:
                        continue
                    f.write(f"__label__good {text}\n")


def train_quality_classifier():
    model = fasttext.train_supervised(input="data/train_quality_classifier.txt")
    model.save_model("data/quality_classifier.bin")


def classify_quality(text: str) -> tuple[str, float]:
    model = fasttext.load_model("data/quality_classifier.bin")
    text = text.replace("\n", " ")
    labels, probabilities = model.predict(text)
    return labels[0].replace("__label__", ""), probabilities[0]


def main(warc_file_path: str):
    # create_training_data(warc_file_path)
    # print("Training data created in 'data/train_quality_classifier.txt'")
    # train_quality_classifier()
    # print("Quality classifier trained and saved to 'data/quality_classifier.bin'")
    pass


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python quality_classifier.py <warc_file_path>")
        sys.exit(1)
    warc_file_path = sys.argv[1]
    main(warc_file_path)
