import sys

from fastwarc.warc import ArchiveIterator, WarcRecordType
import nltk

from cs336_data.extract_text import extract_text


def gopher_quality_filters(text: str):
    lines = text.split("\n")
    count_words = 0
    sum_words_length = 0
    cound_words_alphabetic = 0
    count_valid_lines = 0
    for line in lines:
        if not line.endswith("..."):
            count_valid_lines += 1
        words = nltk.word_tokenize(line)
        count_words += len(words)
        for word in words:
            if sum(c.isalpha() for c in word) >= 1:
                cound_words_alphabetic += 1
            sum_words_length += len(word)
    if count_words < 50 or count_words > 100_000:
        return False
    mean_words_length = sum_words_length / count_words if count_words > 0 else 0
    if mean_words_length < 3 or mean_words_length > 10:
        return False
    invalid_lines_ratio = 1 - (count_valid_lines / len(lines)) if len(lines) > 0 else 0
    if invalid_lines_ratio > 0.3:
        return False
    if cound_words_alphabetic / count_words < 0.6:
        return False
    return True


def main(warc_file_path: str):
    with open(warc_file_path, "rb") as warc_file:
        k = 0
        for record in ArchiveIterator(warc_file):
            if record.record_type == WarcRecordType.response:
                html_bytes = record.reader.read()
                text = extract_text(html_bytes)
                text = text.replace("\n", " ")
                is_good_quality = gopher_quality_filters(text)
                if not is_good_quality:
                    record = text[:100]
                    print("=" * 20)
                    print(f"Record {record}")
                    k += 1
            if k >= 20:
                break


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python gopher_quality_filters.py <warc_file_path>")
        sys.exit(1)
    warc_file_path = sys.argv[1]
    main(warc_file_path)
