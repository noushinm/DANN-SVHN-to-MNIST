import math
import torch

def adjust_lambda(itr, epoch, no_itrs_per_epoch, n_epochs):
    """Calculate lambda parameter for domain adaptation."""
    p = (itr + epoch * no_itrs_per_epoch) / (n_epochs * no_itrs_per_epoch)
    gamma = 10
    return (2 / (1 + math.exp(-gamma * p))) - 1

def evaluate_model(model, dataloader, device):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total