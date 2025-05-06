from torch import nn
import torch
from tqdm import tqdm
import numpy as np

class MLP:
    def __init__(self,
                 input_size: int,
                hidden_size: int,
                output_size: int):

        layers = [
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        ]
        self.model = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        return self.model(x)

    def train(self, X_train: np.ndarray,
                    Y_train: np.ndarray,
                    max_epochs: int = 200,
                    lr: float = 0.001,
                    batch_size: int = 32):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()
        
        X = torch.FloatTensor(X_train)
        y = torch.FloatTensor(Y_train)

        # Create progress bar for epochs
        pbar = tqdm(range(max_epochs), desc='Training Progress')

        for epoch in pbar:
            # Shuffle the data at the start of each epoch
            indices = torch.randperm(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0.0
            num_batches = 0

            for i in range(0, len(X), batch_size):
                batch_X = X_shuffled[i:i + batch_size]
                batch_y = y_shuffled[i:i + batch_size]

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            # Calculate average loss for the epoch
            avg_epoch_loss = epoch_loss / num_batches

            # Update progress bar with current loss
            pbar.set_postfix({'loss': f'{avg_epoch_loss:.6f}'})

    def predict(self, X_test_arr: np.ndarray) -> np.ndarray:
        X_test_tensor = torch.tensor(X_test_arr, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            Y_pred_tensor = self.model(X_test_tensor)
            Y_pred_arr = Y_pred_tensor.cpu().numpy()
        return Y_pred_arr