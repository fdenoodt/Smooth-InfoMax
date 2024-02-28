from options import get_options
import matplotlib.pyplot as plt
import numpy as np

opt = get_options()


def r(val):
    return np.round(val, 4)


class EncoderStatistics:
    def __init__(self, train_losses: np.ndarray[np.float64, np.float64],
                 val_losses: np.ndarray[np.float64, np.float64]):
        self.train_losses = train_losses
        self.val_losses = val_losses

    def __str__(self):
        res = f"Encoder \n"
        for i, (train, val) in enumerate(zip(self.train_losses, self.val_losses)):
            res += f"Module {i} Train Loss: {r(train[-1])}, Val Loss: {r(val[-1])}\n"

        return res


class DownstreamTaskStatistics:
    def __init__(self, accuracy: np.float64, final_accuracy: np.float64,
                 train_loss: np.ndarray[np.float64], val_loss: np.ndarray[np.float64]):
        self.accuracy = accuracy
        self.final_accuracy = final_accuracy
        self.train_loss = np.reshape(train_loss, (-1, 1))
        self.val_loss = np.reshape(val_loss, (-1, 1))

    def __str__(self):
        return f"Downstream task \nAccuracy: {r(self.accuracy)}, Final Accuracy: {r(self.final_accuracy)}, " \
               f"Train Loss: {r(self.train_loss[-1][-1])}, Val Loss: {r(self.val_loss[-1][-1]) if len(self.val_loss) > 0 else 'N/A'}"


def load_loss_npy() -> EncoderStatistics:
    path = opt.log_path
    train_loss = np.load(path + "/train_loss.npy", allow_pickle=True)
    val_loss = np.load(path + "/val_loss.npy", allow_pickle=True)

    return EncoderStatistics(train_loss, val_loss)


def load_downstreamtask_npy(task: str) -> DownstreamTaskStatistics:
    assert task in ["linear_model_speaker", "linear_model_phones"]

    path = f"{opt.log_path}/{task}/"

    accuracy = np.load(path + "/accuracy.npy", allow_pickle=True)
    final_accuracy = np.load(path + "/final_accuracy.npy", allow_pickle=True)
    train_loss = np.load(path + "/train_loss.npy", allow_pickle=True)
    val_loss = np.load(path + "/val_loss.npy", allow_pickle=True)

    return DownstreamTaskStatistics(accuracy, final_accuracy, train_loss, val_loss)


def plot_loss(train_loss, val_loss):
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.show()


def print_statistics(encoder: EncoderStatistics, speaker: DownstreamTaskStatistics, phones: DownstreamTaskStatistics):
    print(encoder)

    print("\nSpeaker:")
    print(speaker)

    print("\nPhones:")
    print(phones)



if __name__ == "__main__":
    e_stat = load_loss_npy()
    s_stat = load_downstreamtask_npy("linear_model_speaker")
    p_stat = load_downstreamtask_npy("linear_model_phones")

    print_statistics(e_stat, s_stat, p_stat)
