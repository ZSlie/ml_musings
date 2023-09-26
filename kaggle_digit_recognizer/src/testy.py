from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import pandas as pd



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # in_channels: int,
        # out_channels: int,
        # kernel_size: _size_2_t,
        # stride: _size_2_t = 1,
        # padding: _size_2_t | str = 0,
        # dilation: _size_2_t = 1,
        # groups: int = 1,
        # bias: bool = True,
        # padding_mode: str = 'zeros',
        # device: Any | None = None,
        # dtype: Any | None = None 
        input_size = 28
        kernel_size = 5
        kernel_size2 = 5
        out_channel2 = 64
        h_w = (input_size, input_size)
        kernel_sizeT = kernel_size
        h, w = conv_output_shape(h_w, kernel_sizeT)

        # print("Calculated H: ", h, " W: ", w)
        h2, w2 = conv_output_shape((h,w), (kernel_size2, kernel_size2))
        # print("Calculated H2: ", h2, " W2: ", w2)
        input_channels = 1 # because grayscale
        self.conv1 = nn.Conv2d(input_channels, h, kernel_size, 1)
        self.conv2 = nn.Conv2d(h, h2, kernel_size2, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        in_features = 2000 # 64 * 28 * 28 * 2 
        # print("In features: " , in_features)
        out_features1 = 24 
        # print("OF 1 : ", out_features1)
        self.fc1 = nn.Linear(int(in_features), out_features1)
        self.fc2 = nn.Linear(out_features1, 10)

    def forward(self, x):
        # # print("Input shape: ", x.shape)
        x = self.conv1(x)
        x = F.relu(x)
        # print("Shape after conv 1: ", x.shape)
        x = self.conv2(x)
        x = F.relu(x)
        # print("Shape after conv2: ", x.shape)
        x = F.max_pool2d(x, 2)
        # print("Shape after max pool2d: ", x.shape)
        x = self.dropout1(x)
        # print("Shape after dropout1: ", x.shape)
        x = torch.flatten(x, 1)
        # print("Shape after flatten: ", x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        # print("Shape after fc1: ", x.shape)
        x = self.dropout2(x)
        # print("Shape after dropout: ", x.shape)
        x = self.fc2(x)
        # print("Shape after fc2: ", x.shape)
        output = F.log_softmax(x, dim=1)
        return output



"""
Utility function for computing output of convolutions
takes a tuple of (h,w) and returns a tuple of (h,w)
"""
def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor( ((h_w[0] + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
    w = floor( ((h_w[1] + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
    return h, w

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    print("finished one training loop")
    for batch_idx, (data, target) in enumerate(train_loader):
        # print("target: ", target)
        # print("target.size: ", target.size())
        # print("Tensor size: ", data.size()) # 64, 1, 28, 28
        data, target = data.to(device), target.to(device)
        # print("Data & Target")
        optimizer.zero_grad()
        # print("0 grad complete")
        output = model(data)
        # print("Output: ", output)
        loss = F.nll_loss(output, target)
        # print("loss: ", loss)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
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
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=21, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args, unknown = parser.parse_known_args()
    print(unknown)
    print(args)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()
    print("Use CUDA: ", use_cuda)
    print("Use MPS: ", use_mps)

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    print("Begging net to device")
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    print("Starting Epochs")
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()
    print("finished Epochs")
    if args.save_model:
        torch.save(model.state_dict(), "../data/mnist_cnn.pt")


if __name__ == '__main__':
    main()
    model = Net()
    model.load_state_dict(torch.load('../data/mnist_cnn.pt'))
    model.eval()  # set to evaluation mode

    data = pd.read_csv('../data/test.csv')
    print("Test data length: ", len(data))
    images = torch.tensor(data.values, dtype=torch.float32)
    images = images.view(-1, 1, 28, 28)

    # Predict using loaded model
    with torch.no_grad():
        predictions = model(images)
        _, predicted_digits = torch.max(predictions, 1)  # get the class (digit) with highest probability

    print(len(predicted_digits))
    print(predicted_digits.numpy())
    predicted_df = pd.DataFrame(predicted_digits.numpy(), columns=['Label'])
    predicted_df.index += 1 
    #predicted_df['ImageId'] = predicted_df.index
    #print(predicted_df)
    predicted_df.to_csv('../data/predicted_digits.csv', index_label='ImageId')