import re
import sys

from fastwarc.warc import ArchiveIterator, WarcRecordType

from cs336_data.extract_text import extract_text


def mask_email(text) -> tuple[str, int]:
    pattern = r"\b[^@\s]+@[^@\s]+\.[a-zA-Z]{2,}\b"
    res = re.subn(pattern, "|||EMAIL_ADDRESS|||", text)
    return res


def mask_phone_number(text) -> tuple[str, int]:
    pattern = re.compile(
        r"""
(?<!\w)
(?:\+?\d{1,3}[\s\.-]*)?
(?:
    \(\d{2,4}\)
    |
    \d{2,4}
)
[\s\.-]*
\d{3,5}
(?:[\s\.-]*\d{3,5}){1,2}
(?!\w)
""",
        re.VERBOSE,
    )
    res = re.subn(pattern, "|||PHONE_NUMBER|||", text)
    return res


def mask_ip_address(text) -> tuple[str, int]:
    pattern = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    res = re.subn(pattern, "|||IP_ADDRESS|||", text)
    return res


def main(warc_file_path: str):
    with open(warc_file_path, "rb") as warc_file:
        k = 0
        for record in ArchiveIterator(warc_file):
            if record.record_type == WarcRecordType.response:
                html_bytes = record.reader.read()
                text = extract_text(html_bytes)
                text = text.replace("\n", " ")
                original_text = text[:100]
                text, n = mask_email(text)
                text, m = mask_phone_number(text)
                text, k = mask_ip_address(text)
                if n > 0 or m > 0 or k > 0:
                    record = original_text[:100]
                    record_modif = text[:100]
                    print("=" * 20)
                    print(f"Record: {record}\nModified Record: {record_modif}")
                    k += 1
            if k >= 20:
                break


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python mask_pii.py <warc_file_path>")
        sys.exit(1)
    warc_file_path = sys.argv[1]
    main(warc_file_path)
