import hashlib


def exact_deduplication(input_file_paths, output_dir):
    log = []

    counter = {}
    for input_file_path in input_file_paths:
        with open(input_file_path, "r") as file:
            lines = file.readlines()

        for line in lines:
            h = hashlib.sha256(line.encode()).hexdigest()

            counter.setdefault(h, 0)
            counter[h] += 1

    for input_file_path in input_file_paths:
        with open(input_file_path, "r") as file:
            lines = file.readlines()

        with open(output_dir / input_file_path.name, "w") as output_file:
            for line in lines:
                h = hashlib.sha256(line.encode()).hexdigest()

                if counter[h] <= 1:
                    output_file.write(line)

            log.append(output_dir / input_file_path.name)

    return log
