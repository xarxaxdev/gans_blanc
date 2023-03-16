import os


# visualization libraries
import matplotlib.pyplot as plt
import numpy as np


### Here will be the main functions to evaluate the models
### Maybe draw some plots in src/plots




def save_plot_train_loss(train_loss,filename):
    epochs = len(train_loss)

    fig, ax = plt.subplots(figsize=(10, 4))
    # visualize the loss values
    ax.plot(train_loss)
    # set the labels
    ax.set_ylabel('Loss')
    ax.set_xlabel(f'{epochs} Epochs')
    fig.tight_layout()

    cur_path = os.path.split(os.path.realpath(__file__))[0]
    datafile = os.path.join(cur_path,'plots',filename)

    plt.savefig(f'{datafile}.png', bbox_inches='tight')
    #plt.show()