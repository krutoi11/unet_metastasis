import torch
import torch.nn as nn
import torch.nn.functional as F


class Unknown(nn.Module):
    def __init__(self, model):
        super(Unknown, self).__init__()

        self.model = model

    def loss(self, true, pred):
        loss = F.mse_loss(true, pred)
        return loss

    def fit(self, trainloader, testloader, epochs=10):
        opt = torch.optim.Adam(lr=3e-3, params=self.model.parameters())

        for _ in range(epochs):
            print("TRAIN")
            for j, batch in enumerate(trainloader):
                data_batch = batch['label']
                label_batch = batch['volume'].float()

                res = self.model(data_batch).squeeze()
                print('Predict', res)
                print('True', label_batch)
                loss = self.loss(res, label_batch)
                opt.zero_grad()
                loss.backward()
                opt.step()

                print(loss.item())

            print("VALIDATION")
            with torch.no_grad():
                for j, batch in enumerate(testloader):
                    data_batch = batch['label']
                    label_batch = batch['volume'].float()

                    res = self.model(data_batch).squeeze()
                    print('Predict', res)
                    print('True', label_batch)
                    loss = self.loss(res, label_batch)
                    print(loss.item())
