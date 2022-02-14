import numpy as np
from sklearn.metrics import f1_score


def f1_score_func(y_true, y_pred, mask):
    """
    Args:
        y_pred: (nbest, batch_size, seq_length). a batch of predicted label_ids -> pred_tags
        y_true: (batch_Size, seq_len). a batch of true label_ids  -> data['label_ids']
        mask: (batch_Size, seq_len). a batch of attention mask
    """
    y_pred = y_pred.squeeze(0)
    seq_ends = (mask.sum(dim=1) - 1).tolist()
    batch_size = mask.shape[0]

    y_true_flat = []
    y_pred_flat = []

    for i, j in zip(np.arange(batch_size), seq_ends):
        print(i, j, '\n')
        print(y_true[i, :j])
        y_true_flat.extend(y_true[i, :j].cpu().numpy())
        y_pred_flat.extend(y_pred[i, :j].cpu().numpy())

    f1 = f1_score(y_true_flat, y_pred_flat, average='weighted')

    return f1