import os


def read_files(path="data/motion/"):
    return list(map(lambda file: path + file, os.listdir(path)))
##input(r"Enter the path of the folder: ")
