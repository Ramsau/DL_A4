# Dependencies
import os
import pickle
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import IMDBDataset

from tqdm import tqdm  # Fancy training progress bar

# Our includes
from vocab import Vocabulary
from model import RNNClassifier
from config import *

from matplotlib import pyplot as plt

def debug_print_sample(model, dataset, vocab, device, config):
    indices = config['print_batches_raw_indices']
    model.eval()
    print("\n--- Debug: Sample Predictions with Raw Text ---")
    for i in indices:
        text, label = dataset[i]
        numericalized = torch.tensor(vocab.numericalize(text), dtype=torch.long)
        length = torch.tensor([len(numericalized)], dtype=torch.long)
        inputs = numericalized.unsqueeze(0).to(device)
        length = length.to(device)
        output = model(inputs, length)
        prediction = torch.sigmoid(output) >= 0.5

        print(f"\nSample index {i}:")
        print(f"Text: {text[:200]}...")  # Only print first 200 characters
        print(f"Prediction: {prediction.item()}, True label: {label}")
        print("---")

def collate_fn(batch, vocab):
    texts, labels = zip(*batch)
    numericalized = [torch.tensor(vocab.numericalize(text), dtype=torch.long) for text in texts]
    lengths = torch.tensor([len(seq) for seq in numericalized], dtype=torch.long)
    padded = pad_sequence(numericalized, batch_first=True, padding_value=vocab.stoi['<pad>'])
    labels = torch.tensor(labels, dtype=torch.float)
    return padded, lengths, labels

def build_vocab_from_dataset(dataset, vocab_file="vocab.pkl"):
    if os.path.exists(vocab_file):
        print(f"Loading vocabulary from {vocab_file}...")
        with open(vocab_file, "rb") as f:
            vocab = pickle.load(f)
    else:
        print("Building vocabulary...")
        texts = [text for text, _ in dataset]
        vocab = Vocabulary(min_freq=2)
        vocab.build_vocab(texts)
        print(f"Vocabulary size: {len(vocab.itos)}. Saving to {vocab_file}...")
        with open(vocab_file, "wb") as f:
            pickle.dump(vocab, f)

    return vocab

def main(config):
    print("\n=======================================")
    print(f"Running with config: {config}")
    print("=======================================\n")

    # Create datasets
    dataset_train = IMDBDataset(train=True)
    dataset_test = IMDBDataset(train=False)

    # Build/load vocabulary
    vocab = build_vocab_from_dataset(dataset_train)

    # Create DataLoaders
    train_loader = DataLoader(
        dataset_train, batch_size=config["batch_size"], shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, vocab)
    )
    test_loader = DataLoader(
        dataset_test, batch_size=config["batch_size"], shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, vocab)
    )

    model = RNNClassifier(
        vocab_size=len(vocab.itos),
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        bidirectional=config['bidirectional']
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_losses = []
    test_losses = []
    best_test_accuracy = 0.0

    for epoch in range(config['num_epochs']):
        print(f"\n=== Starting Epoch {epoch + 1}/{config['num_epochs']} ===")
        model.train()
        total_loss = 0

        for batch_idx, (inputs, lengths, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            inputs, lengths, labels = inputs.to(device), lengths.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_train_loss = total_loss / len(train_loader)
        train_losses.append(average_train_loss)
        print(f"--- Epoch [{epoch + 1}] Average Training Loss: {average_train_loss:.4f} ---")

        model.eval()
        correct = 0
        total = 0
        total_test_loss = 0

        with torch.no_grad():
            for inputs, lengths, labels in test_loader:
                inputs, lengths, labels = inputs.to(device), lengths.to(device), labels.to(device)
                outputs = model(inputs, lengths)
                predictions = (torch.sigmoid(outputs) >= 0.5).float()
                total += labels.size(0)
                correct += (predictions == labels).sum().item()
                test_loss = criterion(outputs, labels)
                total_test_loss += test_loss.item()

        average_test_loss = total_test_loss / len(test_loader)
        test_losses.append(average_test_loss)

        test_accuracy = 100.0 * correct / total
        print(f"Test Accuracy after epoch {epoch + 1}: {test_accuracy:.2f}%")

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab': vocab,
            }, config['save_model_name'] + "_best.pth")

        if config['print_batches_raw']:
            debug_print_sample(model, dataset_test, vocab, device, config)

    final_save_path = config['save_model_name'] + ".pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
    }, final_save_path)
    print(f"\nTraining complete. Best Test Accuracy: {best_test_accuracy:.2f}%")
    print(f"Saved final model to {final_save_path}.\n")

    plt.figure(figsize=(20, 10))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f"Loss Curves for config: {config}")
    plt.show()

    return best_test_accuracy


def run_grid_search():
    configs = get_grid_configs()
    best_acc = 0.0
    best_config = None

    for cfg in configs:
        acc = main(cfg)
        if acc > best_acc:
            best_acc = acc
            best_config = cfg

    print("\n=======================================")
    print("Grid Search Complete!")
    print(f"Best Accuracy: {best_acc:.2f}%")
    print("Best Config:")
    print(best_config)
    print("=======================================")


if __name__ == '__main__':

    # single_config = get_config()
    # main(single_config)

    run_grid_search()
