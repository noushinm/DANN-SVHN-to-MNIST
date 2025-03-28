import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from models.feature_extractor import FeatureExtractor
from models.label_classifier import LabelClassifier
from models.domain_classifier import DomainClassifier
from data.digits import get_mnist, get_svhn
from utils.helpers import adjust_lambda, evaluate_model


def train_source_only(feature_extractor, label_classifier):
    """Train model using source data only"""
    # Initialize optimizer and loss
    optimizer = optim.Adam(
        list(feature_extractor.parameters()) + list(label_classifier.parameters()),
        lr=1e-2,
        betas=(0.9, 0.999),
        weight_decay=0.0005
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

    # Get source data
    src_trX, src_trY, _, _ = get_svhn(getRGB=True)
    src_trX = torch.tensor(src_trX)
    src_trY = torch.tensor(src_trY)

    # Training loop
    BATCH_SIZE = 64
    EPOCHS = 5
    m = src_trX.shape[0]

    feature_extractor.train()
    label_classifier.train()

    for epoch in range(EPOCHS):
        for i in range((m - 1) // BATCH_SIZE + 1):
            # Get batch
            batch_x = src_trX[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
            batch_y = src_trY[i*BATCH_SIZE : (i+1)*BATCH_SIZE]

            if torch.cuda.is_available():
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()

            # Forward pass
            features = feature_extractor(batch_x)
            outputs = label_classifier(features)
            loss = criterion(outputs, batch_y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if i % 200 == 0:
                print(f"Epoch: {epoch+1}/5, iter: {i:4d}, loss: {loss.item():.4f}")



def train_dann(feature_extractor, label_classifier, domain_classifier):
    """Train model using DANN approach"""
    # Initialize optimizers and losses
    optimizer = optim.Adam(
        list(feature_extractor.parameters()) + list(label_classifier.parameters()),
        lr=1e-2,
        betas=(0.9, 0.999),
        weight_decay=0.0005
    )
    optimizer_d = optim.Adam(
        domain_classifier.parameters(),
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=0.0005
    )

    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCELoss()

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

    # Get source and target data
    src_trX, src_trY, _, _ = get_svhn(getRGB=True)
    tgt_trX, tgt_trY, _, _ = get_mnist(getRGB=True, setSizeTo32=False)

    src_trX = torch.tensor(src_trX)
    src_trY = torch.tensor(src_trY)
    tgt_trX = torch.tensor(tgt_trX)

    # Training loop
    BATCH_SIZE = 64
    EPOCHS = 5
    max_len = max(src_trX.shape[0], tgt_trX.shape[0])

    feature_extractor.train()
    label_classifier.train()
    domain_classifier.train()

    for epoch in range(EPOCHS):
        for i in range((max_len - 1) // BATCH_SIZE + 1):
            # Get source and target batches
            src_idx = i % ((src_trX.shape[0] - 1) // BATCH_SIZE + 1)
            tgt_idx = i % ((tgt_trX.shape[0] - 1) // BATCH_SIZE + 1)

            s_x = src_trX[src_idx*BATCH_SIZE : (src_idx+1)*BATCH_SIZE]
            s_y = src_trY[src_idx*BATCH_SIZE : (src_idx+1)*BATCH_SIZE]
            t_x = tgt_trX[tgt_idx*BATCH_SIZE : (tgt_idx+1)*BATCH_SIZE]

            if torch.cuda.is_available():
                s_x = s_x.cuda()
                s_y = s_y.cuda()
                t_x = t_x.cuda()

            # Calculate lambda for gradient reversal
            p = float(i + epoch * max_len) / (EPOCHS * max_len)
            lambda_p = 2. / (1. + np.exp(-10 * p)) - 1

            # Train on source domain
            features_s = feature_extractor(s_x)
            outputs_s = label_classifier(features_s)
            class_loss = class_criterion(outputs_s, s_y)

            # Train domain classifier
            features_t = feature_extractor(t_x)
            domain_s = domain_classifier(features_s, lambda_p)
            domain_t = domain_classifier(features_t, lambda_p)

            domain_labels_s = torch.ones(domain_s.size()).cuda() if torch.cuda.is_available() else torch.ones(domain_s.size())
            domain_labels_t = torch.zeros(domain_t.size()).cuda() if torch.cuda.is_available() else torch.zeros(domain_t.size())

            domain_loss = domain_criterion(domain_s, domain_labels_s) + domain_criterion(domain_t, domain_labels_t)

            loss = class_loss + domain_loss

            # Update parameters
            optimizer.zero_grad()
            optimizer_d.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer_d.step()
            scheduler.step()

            if i % 200 == 0:
                print(f"Epoch: {epoch+1}/5, iter: {i:4d}, lambda: {lambda_p:.2f}, "
                      f"class_loss: {class_loss.item():.4f}, domain_loss: {domain_loss.item():.4f}")



def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['source_only', 'dann'], required=True)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize models
    feature_extractor = FeatureExtractor().to(device)
    label_classifier = LabelClassifier().to(device)
    domain_classifier = DomainClassifier().to(device)

    # Train model based on selected mode
    if args.mode == 'source_only':
        train_source_only(feature_extractor, label_classifier)
    else:
        train_dann(feature_extractor, label_classifier, domain_classifier)

    # Save trained models
    checkpoint = {
        'feature_extractor': feature_extractor.state_dict(),
        'label_classifier': label_classifier.state_dict(),
    }

    if args.mode == 'dann':
        checkpoint['domain_classifier'] = domain_classifier.state_dict()

    torch.save(checkpoint, f'checkpoint_{args.mode}.pth')

if __name__ == '__main__':
    main()
