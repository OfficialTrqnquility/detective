from converter.JsonConverter import decode_file
from scanner.FileScanner import read_files



def main(name):
    dictionaries = [decode_file(*read_files())]



if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
