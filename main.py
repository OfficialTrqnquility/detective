import torch
from torch.utils.data import DataLoader

from model import train
from model.configuration.configuration import ModelConfiguration
from model.data.MovementDataSet import MovementDataSet
from model.model import Model


def main():
    dataset = MovementDataSet("data/motion/")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    print(len(dataloader.dataset.__getitem__(0)[0]))
    model = Model(ModelConfiguration(len(dataloader.dataset.__getitem__(0)[0]), 2)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    loss_fn = torch.nn.BCELoss()
    train.train_model(10, model, device, dataloader, optimizer, loss_fn)


if __name__ == '__main__':
    main()
