import json


def decode_file(file):
    return json.loads(file.read())


def decode_files(files):
    return [decode_file(open(file)) for file in files]
