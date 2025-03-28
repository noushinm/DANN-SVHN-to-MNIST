import argparse
import torch
from models.feature_extractor import FeatureExtractor
from models.label_classifier import LabelClassifier
from models.domain_classifier import DomainClassifier
from data.digits import get_mnist, get_svhn
from utils.helpers import evaluate_model
from torch.utils.data import TensorDataset, DataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', choices=['source_only', 'dann'], required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    args = parser.parse_args()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load models
    feature_extractor = FeatureExtractor().to(device)
    label_classifier = LabelClassifier().to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint_path)
    feature_extractor.load_state_dict(checkpoint['feature_extractor'])
    label_classifier.load_state_dict(checkpoint['label_classifier'])

    # Combine models for evaluation
    class CombinedModel(torch.nn.Module):
        def __init__(self, feature_extractor, label_classifier):
            super(CombinedModel, self).__init__()
            self.feature_extractor = feature_extractor
            self.label_classifier = label_classifier

        def forward(self, x):
            features = self.feature_extractor(x)
            return self.label_classifier(features)

    combined_model = CombinedModel(feature_extractor, label_classifier).to(device)
    combined_model.eval()

    # Load all data (train and test for both domains)
    svhn_train_X, svhn_train_y, svhn_test_X, svhn_test_y = get_svhn(getRGB=True)
    mnist_train_X, mnist_train_y, mnist_test_X, mnist_test_y = get_mnist(getRGB=True, setSizeTo32=True)

    # Create dataloaders for all datasets
    svhn_train_dataset = TensorDataset(
        torch.FloatTensor(svhn_train_X),
        torch.LongTensor(svhn_train_y)
    )
    svhn_test_dataset = TensorDataset(
        torch.FloatTensor(svhn_test_X),
        torch.LongTensor(svhn_test_y)
    )
    mnist_train_dataset = TensorDataset(
        torch.FloatTensor(mnist_train_X),
        torch.LongTensor(mnist_train_y)
    )
    mnist_test_dataset = TensorDataset(
        torch.FloatTensor(mnist_test_X),
        torch.LongTensor(mnist_test_y)
    )

    svhn_train_loader = DataLoader(svhn_train_dataset, batch_size=128, shuffle=False)
    svhn_test_loader = DataLoader(svhn_test_dataset, batch_size=128, shuffle=False)
    mnist_train_loader = DataLoader(mnist_train_dataset, batch_size=128, shuffle=False)
    mnist_test_loader = DataLoader(mnist_test_dataset, batch_size=128, shuffle=False)

    # Evaluate on all datasets
    source_train_acc = evaluate_model(combined_model, svhn_train_loader, device)
    source_test_acc = evaluate_model(combined_model, svhn_test_loader, device)
    target_train_acc = evaluate_model(combined_model, mnist_train_loader, device)
    target_test_acc = evaluate_model(combined_model, mnist_test_loader, device)

    print(f"Model type: {args.model_type}")
    print(f"Source (SVHN) train accuracy: {source_train_acc:.2f}%")
    print(f"Source (SVHN) test accuracy: {source_test_acc:.2f}%")
    print(f"Target (MNIST) train accuracy: {target_train_acc:.2f}%")
    print(f"Target (MNIST) test accuracy: {target_test_acc:.2f}%")

if __name__ == '__main__':
    main()
