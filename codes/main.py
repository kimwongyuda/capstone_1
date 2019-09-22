import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.maxpool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(58 * 58 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        print(x.size())
        x = x.view(-1, 58 * 58 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    print('train start!')

    for batch_idx, (data, target) in enumerate(train_loader):
        print(data.size())
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        print(batch_idx)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    print('test start!')
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    
    #파라미터 입력 부분
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')

    args = parser.parse_args()
    
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()


    torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if use_cuda else "cpu")

    #데이터셋 불러오기
    train_dir = "../data/x-ray/train/"
    test_dir = "../data/x-ray/test/"

    #텐서값 정규화
    normalize = transforms.Normalize((0.1307,),(0.3081,))

    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.Grayscale(),
            transforms.RandomResizedCrop(244),
            transforms.ToTensor(),
            normalize
        ]))

    test_dataset = datasets.ImageFolder(
        test_dir,
        transforms.Compose([
            transforms.Grayscale(),
            transforms.RandomResizedCrop(244),
            transforms.ToTensor(),
            normalize
        ])
    )

    #데이터 로더 불러오기
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size = args.batch_size, shuffle=True,
        num_workers= 4, pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size = args.batch_size, shuffle=False,
        num_workers= 4, pin_memory=True
    )
    
    #모델 만들기
    model = Net().to(device)
    print(model)
    
    #최적화 함수
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    #트레인 및 테스트
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
    
    #모델 저장
    if (args.save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()