from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from gensim.models.poincare import PoincareModel
import pandas as pd
import numpy as np
import torch
import torch.nn as nn


class PhraseEmbeddingDataset(Dataset):
    def __init__(self, X, y, w2v_model, poincare_model, max_len=20):
        self.X = X
        self.y = y
        self.w2v_model = w2v_model
        self.poincare_model = poincare_model
        self.max_len = max_len

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Get Word2Vec embedding
        X = self.get_phrase_vector(self.X.iloc[idx], self.w2v_model, self.max_len)
        
        # Get Poincare embedding
        y = torch.tensor(self.poincare_model.kv[self.y.iloc[idx]], dtype=torch.float)

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
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # 2 for bidirection

    def forward(self, x):
        # Reshape the input to (batch_size, seq_len, features)
        x = x.view(x.size(0), 20, 300)

        # Forward propagate LSTM
        out, _ = self.lstm(x)  # out: tensor of shape (batch_size, seq_length, hidden_size*2)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out



if __name__ == '__main__':
    df = pd.read_csv('/workspaces/master_thesis/mapping/data_ready_to_use.csv')
    df=df.dropna()
    w2v_model = Word2Vec.load("/workspaces/master_thesis/word2vec_pubmed.model")
    poincare_model = PoincareModel.load('/workspaces/master_thesis/poincare_100d_concept_id')
    # Split your phrases into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(df['preprocessed_synonyms_without_stemming'], df['concept_id'], test_size=0.2, random_state=42)

    # Create your datasets
    train_dataset = PhraseEmbeddingDataset(X_train, y_train, w2v_model, poincare_model)
    test_dataset = PhraseEmbeddingDataset(X_test, y_test, w2v_model, poincare_model)

    # Create your data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = BiLSTM(input_size=300, hidden_size=300, output_size=100)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Define the number of epochs
    num_epochs = 50

    # Training loop
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 1000 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))

    # Save the model checkpoint
    torch.save(model.state_dict(), 'model_50epochs_conceptid.ckpt')

