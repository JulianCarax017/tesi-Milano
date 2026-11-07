import torch
from torch import nn

# BERT hidden size (dimensione dell'output dell'encoder)
HIDDEN = 768

# Dimensione layer intermedio dell'MLP
MID = 256

# Probabilità di dropout per regolarizzazione
DROPOUT_PROBABILITY = 0.1


class Classifier(nn.Module):
    def __init__(self, hidden_state=HIDDEN, mid=MID, class_number=2, p=DROPOUT_PROBABILITY):
        super().__init__()

        # MLP: prende il CLS embedding e produce i logits finali
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_state),      # normalizza l'input
            nn.Dropout(p),                   # regolarizzazione
            nn.Linear(hidden_state, mid),    # proiezione a spazio intermedio
            nn.GELU(),                       # attivazione non lineare
            nn.Dropout(p),                   # ulteriore regolarizzazione
            nn.Linear(mid, class_number)     # output logits
        )

    def forward(self, x):
        # x: (batch_size, hidden_state)
        return self.classifier(x)  # logits (batch_size, class_number)
