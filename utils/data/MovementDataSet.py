from torch.utils.data import Dataset


class MovementDataSet(Dataset):

    def __init__(self, data, transform=None):
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
