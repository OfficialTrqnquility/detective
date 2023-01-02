import random

import torch.nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from utils.data.transforms.FlattenArrayTransform import flattenArrayTransform
from utils.data.transforms.RemoveDictKeysTransform import removeDictKeyTransform
from utils.data.transforms.ToTensor import toTensor
from utils.converter.JsonConverter import decode_file
from utils.scanner.FileScanner import read_files

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/motion')


class Net(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Net, self).__init__()

        self.rnn = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.l1 = torch.nn.Linear(in_features=hidden_size, out_features=1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        out, hidden = self.rnn(x)
        out = self.l1(out[:, -1, :])
        # out = self.l2(out)
        # out = self.l3(out)
        out = self.sigmoid(out)
        return out


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


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    test_loss = 0.0
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    positive_targets = 0
    negative_targets = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target.view(output.shape).type(torch.float))
        loss.backward()
        optimizer.step()
        for i in range(len(output)):
            if target[i] == 1:
                positive_targets += 1
                if output[i] > 0.6:
                    true_positives += 1
                else:
                    false_negatives += 1
            elif target[i] == 0:
                negative_targets += 1
                if output[i] < 0.6:
                    true_negatives += 1
                else:
                    false_positives += 1

        if batch_idx % 205 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100.0 * batch_idx / len(train_loader), loss.item()))

    test_loss /= len(train_loader.dataset)

    samples = len(train_loader.dataset)
    correct = true_positives + true_negatives
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)
    writer.add_scalar('training loss', test_loss, epoch)
    writer.add_scalar('accuracy', 100. * correct / samples, epoch)
    writer.add_scalar('Precision', precision, epoch)
    writer.add_scalar('Recall', recall, epoch)
    writer.add_scalar('F1 score', f1_score, epoch)


def test(model, device, test_loader, criterion):
    test_loss = 0.0
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    positive_targets = 0
    negative_targets = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target.view(output.shape).type(torch.float32)).item()

            for i in range(len(output)):
                if target[i] == 1:
                    positive_targets += 1
                    if output[i] > 0.6:
                        true_positives += 1
                    else:
                        false_negatives += 1
                elif target[i] == 0:
                    negative_targets += 1
                    if output[i] < 0.6:
                        true_negatives += 1
                    else:
                        false_positives += 1

    test_loss /= len(test_loader.dataset)
    samples = len(test_loader.dataset)
    correct = true_positives + true_negatives
    wrong = false_positives + false_negatives
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)

    print((wrong / samples))
    print((correct / samples))
    print('''
        Test Set:
        
        Average Loss: {:.4f}
        Accuracy: {}/{} ({:.0f}%)
        Precision: {}/{} ({:.0f}%)
        Recall: {}/{} ({:.0f}%)
        F1 Score: {}
        false positives: {}
        false negatives: {}
        true positives: {}
        true negatives: {}
    
    '''.format(
        test_loss,
        correct, samples, 100. * correct / len(test_loader.dataset),
        true_positives, true_positives + false_positives, 100. * precision,
        true_positives, positive_targets, 100. * recall,
        f1_score,
        false_positives,
        false_negatives,
        true_positives,
        true_negatives

    ))
    # print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
    #       ''.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))


def pack_batch(batch):
    batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    sequences, labels = [], []
    for sequence, label in batch:
        sequences.append(sequence)
        labels.append(label)
    sequences = pad_sequence(sequences, batch_first=True)
    return sequences, torch.tensor(labels)


def preprocessData(data):
    data = removeDictKeyTransform(data)
    data = flattenArrayTransform(data)
    processed = []
    for i in range(3):
        processed.append(data[i] - data[i + 3])
    processed.append(data[4] - data[3])
    processed.append(data[6] - data[5])
    for i in range(7, len(data)):
        processed.append(data[i])

    return data


def main():
    data = []

    for directory in read_files('data/motion/'):
        for file in read_files(directory + '/'):
            file_data = []
            for tick in decode_file(open(file)):
                tick = preprocessData(tick)
                file_data.append(tick)
            data.append((toTensor(file_data), 1 if directory.split("/")[-1] == 'hacking' else 0))

    random.shuffle(data)
    train_data = data[:int((len(data) + 1) * .80)]  # Remaining 80% to training set
    test_data = data[int((len(data) + 1) * .80):]
    train_set = MovementDataSet(train_data)
    test_set = MovementDataSet(test_data)
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, collate_fn=pack_batch)
    test_loader = DataLoader(test_set, batch_size=8, shuffle=True, collate_fn=pack_batch)

    features = len(train_set.__getitem__(0)[0][0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Net(features, 128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    criterion = torch.nn.BCELoss()
    epochs = 500

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)

    test(model, device, test_loader, criterion)

    writer.add_graph(model, iter(test_loader).__next__()[0].to(device))
    writer.close()


if __name__ == '__main__':
    main()
