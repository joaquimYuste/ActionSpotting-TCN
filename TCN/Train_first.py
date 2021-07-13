import torch
print(torch.cuda.is_available())
X_train = torch.FloatTensor([0., 1., 2.])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
X_train = X_train.to(device)

print("hello world")

