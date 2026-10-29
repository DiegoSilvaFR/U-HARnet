"""
Taken from https://github.com/Bjarten/early-stopping-pytorch
"""
import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss
    doesn't improve after a given patience."""

    def __init__(
        self,
        patience=7,
        verbose=False,
        delta=0,
        path="checkpoint.pt",
        trace_func=print,
    ):
        """
        Args:
            patience (int): How long to wait after last time v
                            alidation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each
                            validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity
                            to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            msg = "Validation loss decreased"
            msg = msg + f"({self.val_loss_min:.6f} --> {val_loss:.6f})"
            msg = msg + "Saving model ..."
            self.trace_func(msg)
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

        

"""
Head-only early stopping.

Intended for the 0% fine-tuning regime, where the backbone is frozen
and only model.classifier is trained.
"""

class EarlyStoppingHead:
    """
    Early stops training if validation loss does not improve.

    Only saves the classifier/head weights.

    Intended for experiments where:
        - backbone is frozen
        - only model.classifier is trained
    """

    def __init__(
        self,
        patience=7,
        verbose=False,
        delta=0,
        path="checkpoint_head.pt",
        trace_func=print,
    ):
        """
        Args:
            patience (int):
                Number of epochs to wait after validation loss
                stops improving.

            verbose (bool):
                If True, prints a message whenever the checkpoint
                is updated.

            delta (float):
                Minimum validation-loss improvement required.

            path (str):
                Path where the HEAD-ONLY checkpoint will be saved.

            trace_func (function):
                Function used for printing messages.
        """

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:

            self.best_score = score

            self.save_checkpoint(
                val_loss,
                model,
            )

        elif score < self.best_score + self.delta:

            self.counter += 1

            self.trace_func(
                f"EarlyStopping counter: "
                f"{self.counter} out of {self.patience}"
            )

            if self.counter >= self.patience:
                self.early_stop = True

        else:

            self.best_score = score

            self.save_checkpoint(
                val_loss,
                model,
            )

            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Save ONLY the trained classifier/head.

        The frozen feature extractor is intentionally excluded.
        """

        if not hasattr(model, "classifier"):
            raise AttributeError(
                "Model does not have a 'classifier' attribute."
            )

        if self.verbose:

            msg = (
                "Validation loss decreased "
                f"({self.val_loss_min:.6f} --> {val_loss:.6f}). "
                "Saving HEAD ONLY ..."
            )

            self.trace_func(msg)

        torch.save(
            model.classifier.state_dict(),
            self.path,
        )

        self.val_loss_min = val_loss