import torch
import torch.nn as nn
import torch.optim as optim

# Utility class to calculate running means during model optimization
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
        #Load CPU data and send to GPU
        s0, s1, target = data
        s0 = s0.to(device, dtype=torch.float)
        s1 = s1.to(device, dtype=torch.float)
        target = target.to(device, dtype=torch.long)

        #Net optimization
        with torch.set_grad_enabled(True):
            optimizer.zero_grad()
            output = net(s0, s1)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5) # Had a bit of overfit
            optimizer.step()

        # Stats and console ouput
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
        s0, s1, target = data
        s0 = s0.to(device, dtype=torch.float)
        s1 = s1.to(device, dtype=torch.float)
        target = target.to(device, dtype=torch.long)

        with torch.set_grad_enabled(False):
            output = net(s0, s1)
            loss = criterion(output, target)

        #Also caclulate accuracy metric
        indices = torch.argmax(output, dim=1)
        correct = torch.eq(indices, target).view(-1)
        batch_accuracy = torch.sum(correct).item()/correct.shape[0]
        acc_ra.appendMeasure(batch_accuracy)

        loss_ra.appendMeasure(loss.item())
        if iter_idx!=len(data_loader)-1:
            print('\t Val Iter ',iter_idx+1,' loss ', '{:.5f}'.format(loss_ra.getRunningAverage()),' accuracy ', '{:.5f}'.format(acc_ra.getRunningAverage()), end='\r')
        else:
            print('\t Val Iter ',iter_idx+1,' loss ', '{:.5f}'.format(loss_ra.getRunningAverage()),' accuracy ', '{:.5f}'.format(acc_ra.getRunningAverage()))
