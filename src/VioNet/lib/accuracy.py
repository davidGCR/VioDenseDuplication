import torch

def get_accuracy(y_prob, y_true):
    assert y_true.ndim == 1 and y_true.size() == y_prob.size()

    # print('y_true:', y_true, y_true.size())
    # print('y_prob:', y_prob, y_prob.size())

    y_prob = y_prob >= 0.5
    # print('(y_true == y_prob):', (y_true == y_prob))
    return (y_true == y_prob).sum().item() / y_true.size(0)

def calculate_accuracy_2(y_pred, y_true):
    # Inspired from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html.
    _, predicted = torch.max(y_pred, 1)
    acc = (predicted == y_true).sum().item() / len(y_pred)
    return acc