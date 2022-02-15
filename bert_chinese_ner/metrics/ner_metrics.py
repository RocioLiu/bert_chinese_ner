import numpy as np
from sklearn.metrics import f1_score


def f1_score_func(y_true, y_pred, mask):
    """
    Compute the f1 score of a data_loader
    Args:
        y_pred: (len(data_loader), nbest, batch_size, seq_length).
            a stack of batches of predicted label_ids
        y_true: (len(data_loader), batch_Size, seq_len).
            a stack of batches of true label_ids
        mask: (len(data_loader), batch_Size, seq_len).
            a stack of batches of attention mask
    """

    y_true_flat = []
    y_pred_flat = []

    batches, batch_size = mask.shape[0], mask.shape[1]

    seq_ends_stack = [mask[b].sum(dim=1).tolist() for b in range(batches)]

    for b in range(batches):
        for i, j in zip(np.arange(batch_size), seq_ends_stack[b]):
            y_true_flat.extend(y_true[b][i, :j].cpu().numpy())
            y_pred_flat.extend(y_pred[b].squeeze(0)[i, :j].cpu().numpy())

    f1 = f1_score(y_true_flat, y_pred_flat, average="weighted")

    return f1