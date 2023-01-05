import concurrent.futures
import random

import torch.nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from utils.EarlyStopping import EarlyStopping
from utils.data.transforms.FlattenArrayTransform import flattenArrayTransform
from utils.data.transforms.RemoveDictKeysTransform import removeDictKeyTransform
from utils.data.transforms.ToTensor import toTensor
from utils.converter.JsonConverter import decode_file
from utils.scanner.FileScanner import read_files


class Net(torch.nn.Module):
    def __init__(self, input_size, hidden_size, layers=2):
        super(Net, self).__init__()

        self.rnn = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=layers, batch_first=True)
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

    def evaluateCombination(x):
        layer = x[0]
        neuron = x[1]
        print("Layer: {}, Neuron: {} started".format(layer, neuron))

        results = []
        for i in range(3):

            model = Net(features, neuron, layers=layer).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
            criterion = torch.nn.BCELoss()
            epochs = 200

            es = EarlyStopping()

            for epoch in range(1, epochs + 1):
                train(model, device, train_loader, optimizer, criterion, epoch)
                model.eval()
                vloss = 0.0
                with torch.no_grad():
                    for data, target in test_loader:
                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        vloss += criterion(output, target.view(output.shape).type(torch.float32)).item()
                if es(model, vloss):
                    break

            response = test(model, device, test_loader, criterion)
            if not results:
                results = response
            else:
                results = [a + b for a, b in zip(results, response)]

        results = [i / 3 for i in results]
        print("Layer: {}, Neuron: {} Finished".format(layer, neuron))

        return results[10], layer, neuron, results

    combination = []
    for layer in range(2, 8 ):
        for neuron in range(64, 200):
            combination.append((layer, neuron))
    with concurrent.futures.ThreadPoolExecutor(max_workers=200) as executor:
        results = executor.map(evaluateCombination, combination)
        maximum = max(results, key=lambda x: x[0])
        layer = maximum[1]
        neuron = maximum[2]
        output = maximum[3]
        print('''
                    Results

                    Layers: {}
                    Neurons: {}

                    Average Loss: {:.4f}
                    Accuracy: {:.0f}/{:.0f} ({:.0f}%)
                    Precision: {:.0f}/{:.0f} ({:.0f}%)
                    Recall: {:.0f}/{:.0f} ({:.0f}%)
                    F1 Score: {}
                    false positives: {:.0f}
                    false negatives: {:.0f}
                    true positives: {:.0f}
                    true negatives: {:.0f}
                    '''.format(layer, neuron, *output))


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target.view(output.shape).type(torch.float))
        loss.backward()
        optimizer.step()


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
    precision = (true_positives / (true_positives + false_positives)) if (true_positives != 0) else 0.0
    recall = (true_positives / (true_positives + false_negatives)) if (true_positives != 0) else 0.0
    f1_score = (2 * (precision * recall) / (precision + recall)) if ((precision * recall) != 0) else 0.0

    return [test_loss,
            correct, samples, 100. * correct / len(test_loader.dataset),
            true_positives, true_positives + false_positives, 100. * precision,
            true_positives, positive_targets, 100. * recall,
            f1_score,
            false_positives,
            false_negatives,
            true_positives,
            true_negatives]


if __name__ == '__main__':
    main()
