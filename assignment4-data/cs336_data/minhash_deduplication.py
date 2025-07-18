import hashlib
import os
import random
import re
import string
import unicodedata
import mmh3


def normalize_doc(doc: str) -> str:
    # Lowercasing.
    doc = doc.lower()
    # NFD unicode normalization.
    doc = unicodedata.normalize("NFD", doc)
    # Removing accents and diacritics.
    doc = "".join(char for char in doc if unicodedata.category(char) != "Mn")
    # Removing punctuations.
    doc = doc.translate(str.maketrans("", "", string.punctuation))
    # Normalizing whitespaces.
    doc = re.sub(r"\s+", " ", doc).strip()
    return doc


def find_connected_components_and_pick_random(graph):
    visited = set()

    def dfs(node, component):
        visited.add(node)
        component.append(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs(neighbor, component)

    minhash_signatures = set()
    for node in graph:
        if node not in visited:
            component = []
            dfs(node, component)
            minhash_signature = random.choice(component)
            minhash_signatures.add(minhash_signature)
    return minhash_signatures


def minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    log = []

    minhash_signatures = []
    for input_file_path in input_files:
        with open(input_file_path, "r") as file:
            doc = file.read()
        doc = normalize_doc(doc)
        if doc:
            words = doc.split(" ")
            ngrams_ = [
                str(tuple(words[i : i + ngrams]))
                for i in range(len(words) - ngrams + 1)
            ]
            minhash_signature = tuple(
                min(mmh3.hash(ngram, seed) for ngram in ngrams_)
                for seed in range(num_hashes)
            )
            minhash_signatures.append(minhash_signature)
    matches = {}
    for i in range(len(minhash_signatures)):
        minhash_signature_1 = minhash_signatures[i]
        for j in range(i + 1, len(minhash_signatures)):
            minhash_signature_2 = minhash_signatures[j]
            for r in range(0, num_hashes, num_bands):
                band_1 = minhash_signature_1[r : r + num_bands]
                band_2 = minhash_signature_2[r : r + num_bands]
                if band_1 == band_2:
                    if minhash_signature_1 in matches:
                        matches[minhash_signature_1].append(minhash_signature_2)
                    else:
                        matches[minhash_signature_1] = [minhash_signature_2]
                    break
    true_matches = {}
    identical_signatures = set()
    for minhash_signature_1, matched_signatures in matches.items():
        minhash_signature_1 = minhash_signature_1
        for minhash_signature_2 in matched_signatures:
            minhash_signature_2 = minhash_signature_2
            if minhash_signature_1 == minhash_signature_2:
                identical_signatures.add(minhash_signature_1)
                continue
            collisions = [
                a == b for a, b in zip(minhash_signature_1, minhash_signature_2)
            ]
            jaccard_similarity = sum(collisions) / len(collisions)
            if jaccard_similarity >= jaccard_threshold:
                if minhash_signature_1 not in true_matches:
                    true_matches[minhash_signature_1] = [minhash_signature_2]
                else:
                    true_matches[minhash_signature_1].append(minhash_signature_2)
                if minhash_signature_2 not in true_matches:
                    true_matches[minhash_signature_2] = [minhash_signature_1]
                else:
                    true_matches[minhash_signature_2].append(minhash_signature_1)
    minhash_signatures_to_keep = find_connected_components_and_pick_random(true_matches)
    docs_written = set()
    for input_file_path in input_files:
        with open(input_file_path, "r") as file:
            doc_org = file.read()
        doc = normalize_doc(doc_org)
        if doc:
            words = doc.split(" ")
            ngrams_ = [
                str(tuple(words[i : i + ngrams]))
                for i in range(len(words) - ngrams + 1)
            ]
            minhash_signature = tuple(
                min(mmh3.hash(ngram, seed) for ngram in ngrams_)
                for seed in range(num_hashes)
            )
            if (
                minhash_signature not in true_matches
                and minhash_signature not in docs_written
            ):
                with open(output_directory / input_file_path.name, "w") as output_file:
                    output_file.write(doc_org)
                docs_written.add(minhash_signature)

                log.append(output_directory / input_file_path.name)
            else:
                if minhash_signature in minhash_signatures_to_keep:
                    with open(
                        output_directory / input_file_path.name, "w"
                    ) as output_file:
                        output_file.write(doc_org)
                    docs_written.add(minhash_signature)

                    log.append(output_directory / input_file_path.name)

    return log
