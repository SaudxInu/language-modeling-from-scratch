import sys

from fastwarc.warc import ArchiveIterator, WarcRecordType
import resiliparse.extract
import resiliparse.extract.html2text
import resiliparse.parse


def extract_text(html_bytes: bytes) -> str:
    encoding = resiliparse.parse.encoding.detect_encoding(html_bytes)
    html_str = html_bytes.decode(encoding, "replace")
    text = resiliparse.extract.html2text.extract_plain_text(html_str)
    return text


def main(warc_file_path: str):
    with open(warc_file_path, "rb") as warc_file:
        for record in ArchiveIterator(warc_file):
            if record.record_type == WarcRecordType.response:
                html_bytes = record.reader.read()
                text = extract_text(html_bytes)
                print(text)
                break


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_text.py <warc_file_path>")
        sys.exit(1)
    warc_file_path = sys.argv[1]
    main(warc_file_path)
