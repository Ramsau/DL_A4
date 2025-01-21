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
from config import get_config

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
    print("Creating datasets...")
    dataset_train = IMDBDataset(train=True)
    dataset_test = IMDBDataset(train=False)
    print(f"Number of training samples: {len(dataset_train)}")
    print(f"Number of testing samples: {len(dataset_test)}")

    vocab = build_vocab_from_dataset(dataset_train)

    print("Creating DataLoaders...")
    train_loader = DataLoader(dataset_train, batch_size=config["batch_size"], shuffle=True,
                              collate_fn=lambda batch: collate_fn(batch, vocab))
    # shuffle=True so we not only get 0 in the first test batches to print
    test_loader = DataLoader(dataset_test, batch_size=config["batch_size"], shuffle=True,
                             collate_fn=lambda batch: collate_fn(batch, vocab))

    # Test DataLoader
    print("Testing DataLoader with one batch...")
    try:
        for i, (inputs, lengths, labels) in enumerate(train_loader):
            print(f"Sample Batch {i+1}:")
            print(f"  Inputs shape: {inputs.shape}")
            print(f"  Lengths: {lengths}")
            print(f"  Labels shape: {labels.shape}")
            break
    except Exception as e:
        print(f"Error during DataLoader testing: {e}")
        return

    print("Initializing model, loss function, and optimizer...")
    model = RNNClassifier(vocab_size=len(vocab.itos), embed_dim=config['embed_dim'], hidden_dim=config['hidden_dim']
                          , bidirectional=config['bidirectional'])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    test_losses = []
    train_losses = []
    for epoch in range(config['num_epochs']):
        print(f"\n=== Starting Epoch {epoch+1}/{config['num_epochs']} ===")
        model.train()
        total_loss = 0
        for batch_idx, (inputs, lengths, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
            inputs, lengths, labels = inputs.to(device), lengths.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        print(f"--- Epoch [{epoch + 1}] Average Loss: {average_loss:.4f} ---")
        train_losses.append(average_loss)

        # Evaluation
        print(f"Evaluating after Epoch {epoch + 1}...")
        model.eval()
        correct = 0
        total = 0

        print_batches = config['print_batches_label_amount']
        total_loss = 0
        with torch.no_grad():
            for inputs, lengths, labels in test_loader:
                inputs, lengths, labels = inputs.to(device), lengths.to(device), labels.to(device)
                outputs = model(inputs, lengths)
                predictions = torch.sigmoid(outputs) >= 0.5
                total += labels.size(0)
                correct += (predictions.float() == labels).sum().item()
                loss = criterion(outputs, labels)
                total_loss += loss.item()


                # Prints patches and according predictions
                if print_batches > 0:
                    print(f"\n--- Test Batch {print_batches + 1} ---")
                    print(f"Predictions: {predictions.cpu().numpy()}")
                    print(f"True labels: {labels.cpu().numpy()}")
                    print_batches -= 1
        print(f"Test Accuracy after epoch {epoch + 1}: {100 * correct / total:.2f}%")
        average_loss = total_loss / len(test_loader)
        test_losses.append(average_loss)

        # Also prints patches and predictions, but also raw text...
        if config['print_batches_raw']:
            debug_print_sample(model, dataset_test, vocab, device, config)

    print("Training complete.")

    # Save the trained model along with the vocabulary
    save_path = config['save_model_name'] + ".pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
    }, save_path)
    print("Saved model.")

    plt.figure(figsize=(20, 10))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


if __name__ == '__main__':
    config = get_config()
    main(config)
