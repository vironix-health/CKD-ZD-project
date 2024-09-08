import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import numpy as np
import pickle

# Define Fully Connected Neural Network
class FCNN(nn.Module):
    def __init__(self, input_dim):
        super(FCNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        return x
    
# Define single residual block for tabular data
class ResNetBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ResNetBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        residual = x  # Save the input for the residual connection
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = out + residual  # Add the residual (input) to the output
        out = self.relu(out)
        return out

# Define ResNet model for tabular data
class ResNetForTabular(nn.Module):
    def __init__(self, input_dim, num_blocks, hidden_dim, num_classes):
        super(ResNetForTabular, self).__init__()
        self.initial_fc = nn.Linear(input_dim, hidden_dim)  # Initial fully connected layer
        self.res_blocks = nn.Sequential(
            *[ResNetBlock(hidden_dim, hidden_dim) for _ in range(num_blocks)]  # Stack multiple residual blocks
        )
        self.final_fc = nn.Linear(hidden_dim, num_classes)  # Output layer with num_classes outputs
    
    def forward(self, x):
        out = self.initial_fc(x)
        out = self.res_blocks(out)
        out = self.final_fc(out)
        return out


class DeepLearningWrapper:
    def __init__(self, cfg, input_dim):
        self.cfg = cfg
        self.model = FCNN(input_dim) if self.cfg['tag'] == 'fcnn' else ResNetForTabular(input_dim, self.cfg['num_blocks'], self.cfg['hidden_dim'], self.cfg['num_classes'])
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
        self.best_val_loss = float('inf')

        # Define and move model to device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    # -----------------------------------------------------      TRAINER      ----------------------------------------------------- #  

    def Trainer(self, train_loader, val_loader):
        early_stop_count = 0

        for epoch in range(self.cfg['epochs']):
            self.model.train()
            running_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                if self.cfg['tag'] == 'rsnt':
                    targets = targets.squeeze().float()

                self.optimizer.zero_grad()
                outputs = self.model(inputs) if self.cfg['tag'] == 'fcnn' else self.model(inputs).squeeze()
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            avg_train_loss = running_loss / len(train_loader)
            avg_val_loss = self._validate(val_loader)

            print(f"Epoch {epoch+1}/{self.cfg['epochs']}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}")

            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                early_stop_count = 0
            else:
                early_stop_count += 1

            if early_stop_count >= self.cfg['patience']:
                print("Early stopping triggered")
                break

        # Save model to file
        torch.save(self.model.state_dict(), f"models/{self.cfg['tag']}_MIMIC.pth")

    def _validate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                if self.cfg['tag'] == 'rsnt':
                    labels = labels.squeeze().float()
                outputs = self.model(inputs) if self.cfg['tag'] == 'fcnn' else self.model(inputs).squeeze()
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
        return val_loss / len(val_loader)

    # -----------------------------------------------------      EVALUATOR      ----------------------------------------------------- #

    def Evaluator(self, test_loader):
        self.model.eval()
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                predicted = (outputs > 0).int()
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        auc = roc_auc_score(all_labels, all_predictions)
        print(f"Test AUC: {auc}")
        return auc
