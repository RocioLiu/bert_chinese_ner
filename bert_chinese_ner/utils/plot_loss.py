import matplotlib.pyplot as plt


def loss_f1_plot(history, epochs):
    train_loss = history['train_loss']
    eval_loss = history['eval_loss']
    eval_f1 = history['eval_f1']
    steps = history['step']

    show_every = len(steps) // epochs
    sparse_epochs = [None] * len(steps)
    sparse_epochs[show_every-1::show_every] = list(range(1, epochs+1))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].plot(steps, train_loss, label='Training')
    axes[0].plot(steps, eval_loss, label='Evaluation')
    axes[0].set_xticks(steps)
    axes[0].set_xticklabels(sparse_epochs)

    axes[0].legend(loc='upper right')
    # axes[0].set_xlabel('Epochs', fontsize='large')
    axes[0].set_ylabel('Loss', fontsize='large')

    axes[1].plot(steps, eval_f1)
    axes[1].set_xticks(steps)
    axes[1].set_xticklabels(sparse_epochs)

    axes[1].set_xlabel('Epochs', fontsize='large')
    axes[1].set_ylabel('F1-score', fontsize='large')

    axes[0].set_title('Loss  &  F1-score', fontsize='x-large')