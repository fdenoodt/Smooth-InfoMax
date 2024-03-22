# %%
"""
A folder is given which contains the following pairs of files: train_loss_{idx}.csv and val_loss_{idx}.csv
The script iterates over all {idx} file pairs, loads in the data and generates a graph for the data.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

try:
    import tikzplotlib
except:
    print("tikzplotlib not installed, will not save loss as tex")

# def draw_loss_curve(self, train_loss, val_loss):
#     # assert len(train_loss) == len(val_loss)

#     lst_iter = np.arange(len(train_loss))
#     plt.plot(lst_iter, np.array(train_loss), "-b", label="train loss")

#     lst_iter = np.arange(len(val_loss))
#     plt.plot(lst_iter, np.array(val_loss), "-r", label="val loss")

#     plt.xlabel("epoch")
#     plt.ylabel("loss")
#     plt.legend(loc="upper right")

#     # save image
#     # plt.savefig(os.path.join(self.logging_path, "loss.png"))
#     # plt.close()
#     plt.show()


root = r"D:\thesis_logs\logs\GIM_DECODER_simple_v2_experiment\MEL_SPECTR_n_fft=4096 !!\lr_0.0050000\GIM_L1"
root = r"D:\thesis_logs\logs\xxxx_TWO_MODULE_V3_dim32_kld_weight=0.0033 !!\DECODER\MEL_SPECTR_n_fft=4096\lr_0.0050000\GIM_L1"

# iterate over all files in the folder
for file in os.listdir(root):
    if file.endswith(".csv") and file.startswith("train_loss"):  # first starts with checks VGIM, second checks decoder
        # get the index of the file
        idx = file.split("_")[2].split(".")[0]
        # load in the train and val loss
        train_loss = np.loadtxt(os.path.join(root, f"train_loss_{idx}.csv"))
        val_loss = np.loadtxt(os.path.join(root, f"val_loss_{idx}.csv"))
    elif file.endswith(".csv") and file.startswith("training_loss"):
        train_loss = np.loadtxt(os.path.join(root, "training_loss.csv"))
        val_loss = np.loadtxt(os.path.join(root, "validation_loss.csv"))
        idx = "0"
    else:
        continue

    # draw the loss curve
    # draw_loss_curve(train_loss, val_loss)
    lst_iter = np.arange(len(train_loss))
    plt.plot(lst_iter, np.array(train_loss), "-b", label="train loss")

    lst_iter = np.arange(len(val_loss))
    plt.plot(lst_iter, np.array(val_loss), "-r", label="val loss")

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(loc="upper right")

    # save image
    # plt.savefig(os.path.join(self.logging_path, "loss.png"))
    # plt.close()

    # grid on
    plt.grid(False)

    # save image
    plt.savefig(os.path.join(root, f"loss_{idx}.png"))
    try:
        tikzplotlib.save(os.path.join(root, f"loss_{idx}.tex"))
    except:
        pass
    plt.show()

    # plt.close()
