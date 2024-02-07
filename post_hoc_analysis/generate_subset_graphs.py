try:
    import tikzplotlib
except:
    print("tikzplotlib not installed, will not save loss as tex")
import os
import matplotlib.pyplot as plt

from utils.helper_functions import create_log_dir


"""
The folder root_logs contains folders of the following structure:
linear_model_kld=0.0033_subset=1_epoch=790,
  where kld=0.0033 or kld=0 and
  subset=1, 2, 4 .. 128, "all"

Each folder contains a file: validation_accuracy.csv, which contains the validation accuracy for each epoch.
An example row is:
1.249999952316283824e+01

This script will plot the final validation accuracy for each subset size. (aka the last row of each file)
The plot will contain two lines, one for kld=0.0033 and one for kld=0.
"""

# root_logs = r"D:\thesis_logs\logs\good models\CLASSIFIER_module_2"
# root_save = r"D:\thesis_logs\logs\good models\CLASSIFIER_module_2\plots"
root_logs = r"D:\thesis_logs\logs\good models\generalisation_experiment\module1"
root_save = r"D:\thesis_logs\logs\good models\generalisation_experiment\module1\plots"

# get all folders
folders = os.listdir(root_logs)
folders = [f for f in folders if os.path.isdir(os.path.join(root_logs, f))]
folders = [f for f in folders if "subset" in f]

# get all subset sizes
klds = [(f.split("=")[1]).split("_")[0] for f in folders]
subset_sizes = [(f.split("=")[2]).split("_")[0] for f in folders]

# replace "all" with 128
subset_sizes = [128 * 2 if s == "all" else s for s in subset_sizes]

# convert to int
subset_sizes = [int(s) for s in subset_sizes]

# sort by subset size
subset_sizes, klds, folders = zip(*sorted(zip(subset_sizes, klds, folders)))

# get validation accuracy for each folder
val_acc = []
for folder in folders:
    path = os.path.join(root_logs, folder,
                        r"CrossEntrop Loss\lr_0.0010000\GIM_L1")
    with open(os.path.join(path, "validation_accuracy.csv")) as f:
        lines = f.readlines()
        val_acc.append(float(lines[-1]))

# dividie val_acc by 100 to get percentage
val_acc = [v / 100 for v in val_acc]

# Separate kld=0 and kld=0.0033
val_acc_0 = [val_acc[i] for i in range(len(val_acc)) if klds[i] == "0"]
val_acc_0_0033 = [val_acc[i]
                  for i in range(len(val_acc)) if klds[i] == "0.0035"]

subset_sizes = subset_sizes[::2]

# filter out last subset_size
# subset_sizes = subset_sizes[:-1]
# val_acc_0 = val_acc_0[:-1]
# val_acc_0_0033 = val_acc_0_0033[:-1]

val_acc_random = [1/9 for _ in range(len(subset_sizes))]


# plot
plt.plot(subset_sizes, val_acc_0, "-b", label="kld=0")
plt.plot(subset_sizes, val_acc_0_0033, "-g", label="kld=0.0035")
plt.plot(subset_sizes, val_acc_random, "-r", label="random")

plt.xlabel("Number of datapoints per class (9 classes)")
plt.ylabel("Validation accuracy")
plt.legend()
plt.title("Validation accuracy for different subset sizes")

# log scale
plt.xscale("log")

# x ticks
plt.xticks(subset_sizes, subset_sizes)

# save
create_log_dir(root_save)
plt.savefig(os.path.join(root_save, "validation_accuracy_subset_size.png"))
try:
  tikzplotlib.save(os.path.join(root_save, "validation_accuracy_subset_size.tex"))
except:
   pass


# plt.show()
