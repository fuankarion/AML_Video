import torch
import torch.nn as nn
import torch.optim as optim

class runningAverages():
    def __init__(self):
        self.runningValue = 0
        self.count = 0

    def appendMeasure(self, measure):
        self.runningValue = self.runningValue + measure
        self.count = self.count + 1

    def getRunningValue(self):
        return self.runningValue

    def getRunningAverage(self):
        return self.runningValue/self.count


def train_phase(net, device, data_loader, optimizer, criterion):
    loss_ra = runningAverages()
    net.train()
    for iter_idx, data in enumerate(data_loader):
        s0, s1, s2, target = data
        s0 = s0.to(device, dtype=torch.float)
        s1 = s1.to(device, dtype=torch.float)
        s2 = s2.to(device, dtype=torch.float)
        target = target.to(device, dtype=torch.long)

        with torch.set_grad_enabled(True):
            optimizer.zero_grad()
            output = net(s0, s1, s2)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()

        loss_ra.appendMeasure(loss.item())
        if iter_idx!=len(data_loader)-1:
            print('\t Train Iter ',iter_idx+1,' loss ', '{:.5f}'.format(loss_ra.getRunningAverage()), end='\r')
        else:
            print('\t Train  Iter ',iter_idx+1,' loss ', '{:.5f}'.format(loss_ra.getRunningAverage()))


def test_phase(net, device, data_loader, criterion):
    loss_ra = runningAverages()
    acc_ra = runningAverages()
    net.eval()

    for iter_idx, data in enumerate(data_loader):
        s0, s1, s2, target = data
        s0 = s0.to(device, dtype=torch.float)
        s1 = s1.to(device, dtype=torch.float)
        s2 = s2.to(device, dtype=torch.float)
        target = target.to(device, dtype=torch.long)

        with torch.set_grad_enabled(False):
            output = net(s0, s1, s2)
            loss = criterion(output, target)

        #Accuracy
        indices = torch.argmax(output, dim=1)
        correct = torch.eq(indices, target).view(-1)
        batch_accuracy = torch.sum(correct).item()/correct.shape[0]
        acc_ra.appendMeasure(batch_accuracy)

        # print statistics
        loss_ra.appendMeasure(loss.item())
        if iter_idx!=len(data_loader)-1:
            print('\t Val Iter ',iter_idx+1,' loss ', '{:.5f}'.format(loss_ra.getRunningAverage()),' accuracy ', '{:.5f}'.format(acc_ra.getRunningAverage()), end='\r')
        else:
            print('\t Val Iter ',iter_idx+1,' loss ', '{:.5f}'.format(loss_ra.getRunningAverage()),' accuracy ', '{:.5f}'.format(acc_ra.getRunningAverage()))
