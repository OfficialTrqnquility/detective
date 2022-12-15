import torch
from torch.utils.data import Dataset

from utils.converter.JsonConverter import decode_file
from utils.scanner.FileScanner import read_files


class MovementDataSet(Dataset):

    def __init__(self, root_folder, transform=None):
        data = []

        for directory in read_files(root_folder):
            for file in read_files(directory + '/'):
                for tick in decode_file(open(file)):
                    data.append((self.map_data(tick), directory.split("/")[-1]))

        self.data = data
        self.transform = transform

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, index):
        # Get the data point and label at the specified index
        item, label = self.data[index]

        # Apply the transform to the data point, if provided
        if self.transform:
            item = self.transform(item)


        return item, label

    def map_data(self, data):
        # Initialize the mapped list
        mapped = []

        # Loop through the items in the data dictionary
        for key, value in data.items():
            # If the value is a dictionary, convert its values to a tensor and add it to the mapped list
            if isinstance(value, dict):
                mapped.append(torch.Tensor(list(value.values())))
            # Otherwise, convert the value to a tensor and add it to the mapped list
            else:
                mapped.append(torch.tensor(value))

        # Return the mapped list
        return mapped
