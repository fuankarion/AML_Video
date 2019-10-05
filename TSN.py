import torch
import torch.nn as nn
import torch.optim as optim
from net import resnet_tsn
from optimization import train_phase, test_phase
from datasets import VideoDataset, VideoDatasetVal, transform_train, transform_val

classez = ['Archery', 'BalanceBeam', 'BaseballPitch', 'BenchPress', 'Biking',
'Bowling', 'Fencing', 'HammerThrow', 'HighJump', 'JavelinThrow', 'Kayaking',
'LongJump', 'PoleVault', 'PommelHorse', 'Rowing', 'SkateBoarding', 'SkyDiving',
'SumoWrestling', 'Surfing', 'TrampolineJumping']

net = resnet_tsn(pretrained=True, progress=True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

train_set = VideoDataset('/home/jcleon/tmp/train', transform_train, classez)
val_set = VideoDatasetVal('/home/jcleon/tmp/val', transform_val, classez)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=24, shuffle=True,
                                          num_workers=4)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=24, shuffle=False,
                                          num_workers=4)

has_cuda = torch.cuda.is_available()
device = torch.device('cuda:3' if has_cuda else 'cpu')
net = net.to(device)

for epoch in range(100):
    scheduler.step()
    print('Epoch ', epoch+1)
    train_phase(net, device, train_loader, optimizer, criterion)
    test_phase(net, device, val_loader, criterion)
print('Finished Training')
