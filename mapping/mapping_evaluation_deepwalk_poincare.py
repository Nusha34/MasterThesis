from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from gensim.models.poincare import PoincareModel
import pandas as pd
import numpy as np
import torch
import torch.nn as nn


class PhraseEmbeddingDataset(Dataset):
    def __init__(self, X, y, w2v_model, deepwalk_model, poincare_model, max_len=20):
        self.X = X
        self.y = y
        self.w2v_model = w2v_model
        self.deepwalk_model = deepwalk_model
        self.poincare_model = poincare_model
        self.max_len = max_len

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Get Word2Vec embedding
        X = self.get_phrase_vector(self.X.iloc[idx], self.w2v_model, self.max_len)

        # Get DeepWalk and Poincare embeddings
        deepwalk_embedding = self.deepwalk_model.wv[str(self.y.iloc[idx])]
        poincare_embedding = self.poincare_model.kv[self.y.iloc[idx]]

        # Concatenate embeddings
        y = np.concatenate((deepwalk_embedding, poincare_embedding), axis=0)
        y = torch.tensor(y, dtype=torch.float)

        return X, y

    @staticmethod
    def get_phrase_vector(phrase, model, max_len):
        words = str(phrase).split()
        phrase_vector = np.zeros((max_len, model.vector_size))

        for i in range(max_len):
            if i < len(words) and words[i] in model.wv:
                phrase_vector[i] = model.wv[words[i]]

        phrase_vector = phrase_vector.flatten()

        return torch.tensor(phrase_vector, dtype=torch.float)


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size, hidden_size, batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, output_size)  # 2 for bidirection

    def forward(self, x):
        # Reshape the input to (batch_size, seq_len, features)
        x = x.view(x.size(0), 20, 300)

        # Forward propagate LSTM
        out, _ = self.lstm(
            x
        )  # out: tensor of shape (batch_size, seq_length, hidden_size*2)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


def hyporbolic_distance(x, y):
    # calculate hyporbolic distance between two vectors
    return np.arccosh(
        1
        + 2
        * np.linalg.norm(x - y) ** 2
        / ((1 - np.linalg.norm(x) ** 2) * (1 - np.linalg.norm(y) ** 2))
    )


if __name__ == "__main__":
    df = pd.read_csv("/workspaces/master_thesis/mapping/data_ready_to_use.csv")
    df = df.dropna()
    w2v_model = Word2Vec.load("/workspaces/master_thesis/word2vec_pubmed.model")
    deepwalk_model = Word2Vec.load("/workspaces/master_thesis/deepwalk_snomed.model")
    poincare_model = PoincareModel.load(
        "/workspaces/master_thesis/poincare_100d_concept_id"
    )
    # Split your phrases into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        df["preprocessed_synonyms_without_stemming"],
        df["concept_id"],
        test_size=0.02,
        random_state=42,
    )

    # Create your datasets
    train_dataset = PhraseEmbeddingDataset(
        X_train, y_train, w2v_model, deepwalk_model, poincare_model
    )
    test_dataset = PhraseEmbeddingDataset(
        X_test, y_test, w2v_model, deepwalk_model, poincare_model
    )
    print(len(train_dataset))
    print(len(test_dataset))
    # Create your data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # load the model
    model = BiLSTM(input_size=300, hidden_size=300, output_size=200)
    model.load_state_dict(
        torch.load(
            "/workspaces/master_thesis/model_50epochs_conceptid_deepwalk_poincare.ckpt"
        )
    )
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(device)
    model.eval()
    k_values = [1, 5, 10, 20, 50]
    accuracy_values = []

    with torch.no_grad():
        # Load all data into memory
        inputs_all = []
        labels_all = []
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs_all.append(inputs)
            labels_all.append(labels)

        inputs_all = torch.cat(inputs_all)
        labels_all = torch.cat(labels_all)

        # Compute outputs for all data
        outputs_all = model(inputs_all)
        outputs_all = outputs_all.cpu().numpy()
        labels_all = labels_all.cpu().numpy()

        for k in k_values:
            print(k)
            correct = 0
            total = 0
            for i in range(len(outputs_all)):
                distances = []
                for j in range(len(labels_all)):
                    distances.append(hyporbolic_distance(outputs_all[i], labels_all[j]))
                # get the indices of the k nearest neighbors
                indices = np.argsort(distances)[:k]
                # get the labels of the k nearest neighbors
                nearest_neighbors = labels_all[indices]
                # check if the true label is among the k nearest neighbors
                true_label = labels_all[i]
                if true_label in nearest_neighbors:
                    correct += 1
                total += 1
            accuracy = correct / total
            accuracy_values.append(accuracy)

    for i, accuracy in zip(k_values, accuracy_values):
        print(f"Accuracy for k={i}: {accuracy * 100}%")
