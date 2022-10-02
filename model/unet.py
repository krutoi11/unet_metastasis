import torch.nn as nn
import torch
import time
from losses import weighted_bce, iou


class UNet(nn.Module):
    def __init__(self, device, model):
        super(UNet, self).__init__()

        self.loss = weighted_bce
        self.device = device
        self.model = model.to(self.device)

    @torch.no_grad()
    def metrics(self, pred, true):
        confusion_matrix = pred / true

        TP = torch.sum(confusion_matrix == 1).item()
        FP = torch.sum(confusion_matrix == float('inf')).item()
        TN = torch.sum(torch.isnan(confusion_matrix)).item()
        FN = torch.sum(confusion_matrix == 0).item()

        accuracy = (TP + TN) / (TP + FP + TN + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = (2 * precision * recall) / (precision + recall)

        return accuracy, precision, recall, F1

    def fit(self, n_epochs, trainloader, testloader):
        opt = torch.optim.Adam(lr=3e-4, params=self.model.parameters())

        pytorch_total_params = sum(p.numel() for p in self.model.parameters())
        print(f'Model has {pytorch_total_params} params')

        for i in range(n_epochs):
            """
            TRAINING PART
            """
            loss_train = 0
            f1 = 0
            for j, batch in enumerate(trainloader):
                start_time = time.time()

                data_batch1 = batch['image'].to(self.device)
                data_batch2 = batch['label_before'].to(self.device)
                label_batch = batch['label_after'].to(self.device)

                pred_batch = self.model(data_batch1, data_batch2)

                loss = self.loss(pred_batch, label_batch)
                loss_train += loss

                metrics = self.metrics(torch.round(pred_batch), label_batch)
                f1 += metrics[3]

                opt.zero_grad()
                loss.backward()
                opt.step()

                end_time = time.time() - start_time

                print(f'Epoch {i}/{n_epochs}, batch {j}/{len(trainloader)} \n'
                      f'Batch Loss {round(loss.item(), 5)} \n'
                      f'Accuracy {round(metrics[0], 4)} \n'
                      f'Precision {round(metrics[1], 4)} \n'
                      f'Recall {round(metrics[2], 4)} \n'
                      f'F1 {round(metrics[3], 4)} \n'
                      f'Batch time {round(end_time, 4)}')

            """
            VALIDATION PART
            """
            loss_test = 0
            f1_for_all = 0
            with torch.no_grad():
                for j, batch in enumerate(testloader):
                    batch = batch.to(self.device)

                    data_batch1 = batch['image'].to(self.device)
                    data_batch2 = batch['label_before'].to(self.device)
                    label_batch = batch['label_after'].to(self.device)

                    pred_batch = self.model(data_batch1, data_batch2)
                    l = self.loss(pred_batch, label_batch)

                    loss_test += l

                    metrics = self.metrics(torch.round(pred_batch), label_batch)
                    f1_for_all += metrics[3]

                print(f'Train Loss {loss_train / len(trainloader)} \n'
                      f'Validation Loss {loss_test / len(testloader)} \n'
                      f'F1_train {f1 / len(trainloader)} \n'
                      f'F1_test {f1_for_all / len(testloader)}')
