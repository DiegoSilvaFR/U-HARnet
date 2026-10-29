import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

# 2. Define the Linear Classifier Model
class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearClassifier, self).__init__()
        # Applies y = xA^T + b mapping features to class scores
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.linear(x)


def train_test_linear_probe(X_train, X_test, y_train, y_test, u_test, epochs=20):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    INPUT_DIM = X_train.shape[1]
    NUM_CLASSES = np.unique(y_train).shape[0]

    model = LinearClassifier(
        input_dim=INPUT_DIM,
        num_classes=NUM_CLASSES
    ).to(device)

    X_train = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    X_test = torch.as_tensor(X_test, dtype=torch.float32, device=device)
    y_train = torch.as_tensor(y_train, dtype=torch.long, device=device)

    # Class weights based on y_train occurrences
    # Assumes labels are encoded as 0, ..., NUM_CLASSES - 1
    class_counts = torch.bincount(
        y_train, minlength=NUM_CLASSES
    ).float()

    class_weights = (
        class_counts.sum()
        / (NUM_CLASSES * class_counts.clamp_min(1))
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    model.train()

    for epoch in range(epochs):
        logits = model(X_train)
        loss = criterion(logits, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Inference
    model.eval()
    with torch.no_grad():
        raw_logits = model(X_test)
        predicted_class = torch.argmax(raw_logits, dim=1)

    predicted_class = predicted_class.cpu().numpy()
    y_test = np.asarray(y_test)
    correctness = y_test == predicted_class

    macro_f1 = f1_score(
        y_test, predicted_class,
        average="macro",
        zero_division=0
    )

    if u_test.shape[1] > 1:
        results = {}

        for method in u_test.columns:
            _u_test = u_test[method].values

            c_auroc = roc_auc_score(
                correctness, _u_test
            )

            method_name = method.replace("u_ce", "CE")
            results[method_name] = {
                "macro_f1": macro_f1,
                "c_auroc": c_auroc
            }

        return results

    else:
        c_auroc = roc_auc_score(
            correctness, u_test
        )

        return {
            "macro_f1": macro_f1,
            "c_auroc": c_auroc
        }
