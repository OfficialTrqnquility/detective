import json


def decode_file(file):
    return json.loads(file.read())
