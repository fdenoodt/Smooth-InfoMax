import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import copy
try:
    import tikzplotlib
except:
    print("tikzplotlib not installed, will not save loss as tex")

from config_code.config_classes import OptionsConfig


class Logger:
    def __init__(self, opt: OptionsConfig):
        self.opt = opt

        # create log_dir if not exist
        if not os.path.exists(opt.log_path):
            os.makedirs(opt.log_path)

        modules = opt.encoder_config.architecture.modules
        nb_modules = len(modules)
        if opt.validate:
            self.val_loss = [[] for i in range(nb_modules)]
        else:
            self.val_loss = None

        self.train_loss = [[] for i in range(nb_modules)]

        if opt.encoder_config.start_epoch > 0:
            self.loss_last_training = np.load(
                os.path.join(opt.model_path, "train_loss.npy"), allow_pickle=True
            ).tolist()
            self.train_loss[: len(self.loss_last_training)] = copy.deepcopy(
                self.loss_last_training
            )

            if opt.validate:
                self.val_loss_last_training = np.load(
                    os.path.join(opt.model_path, "val_loss.npy"), allow_pickle=True
                ).tolist()
                self.val_loss[: len(self.val_loss_last_training)] = copy.deepcopy(
                    self.val_loss_last_training
                )
            else:
                self.val_loss = None
        else:
            self.loss_last_training = None

            if opt.validate:
                self.val_loss = [[] for i in range(nb_modules)]
            else:
                self.val_loss = None

        self.num_models_to_keep = 1
        assert self.num_models_to_keep > 0, "Dont delete all models!!!"

    def np_save(self, path, data):
        np.save(path, data)
        for idx, item in enumerate(data):
            try: # just for the decoder
                np.savetxt(f"{path}_{idx}.csv", item, delimiter=",")
            except:
                pass

    def create_log(
        self,
        model,
        accuracy=None,
        epoch=0,
        optimizer=None,
        final_test=False,
        final_loss=None,
        acc5=None,
        classification_model=None,
    ):

        print("Saving model and log-file to " + self.opt.log_path)

        # Save the model checkpoint
        if self.opt.experiment == "vision":
            for idx, layer in enumerate(model.module.encoder):
                torch.save(
                    layer.state_dict(),
                    os.path.join(
                        self.opt.log_path, "model_{}_{}.ckpt".format(idx, epoch)
                    ),
                )
        elif self.opt.experiment == "RMSE_decoder":
            torch.save(
                model.state_dict(),
                os.path.join(self.opt.log_path, "model_{}.ckpt".format(epoch)),
            )
            
        else:
            torch.save(
                model.state_dict(),
                os.path.join(self.opt.log_path, "model_{}.ckpt".format(epoch)),
            )

        ### remove old model files to keep dir uncluttered
        if (epoch - self.num_models_to_keep) % 10 != 0:
            try:
                if self.opt.experiment == "vision":
                    for idx, _ in enumerate(model.module.encoder):
                        os.remove(
                            os.path.join(
                                self.opt.log_path,
                                "model_{}_{}.ckpt".format(
                                    idx, epoch - self.num_models_to_keep
                                ),
                            )
                        )
                else:
                    os.remove(
                        os.path.join(
                            self.opt.log_path,
                            "model_{}.ckpt".format(epoch - self.num_models_to_keep),
                        )
                    )
            except:
                print("not enough models there yet, nothing to delete")

        if classification_model is not None:
            # Save the predict model checkpoint
            torch.save(
                classification_model.state_dict(),
                os.path.join(
                    self.opt.log_path, "classification_model_{}.ckpt".format(epoch)
                ),
            )

            ### remove old model files to keep dir uncluttered
            try:
                os.remove(
                    os.path.join(
                        self.opt.log_path,
                        "classification_model_{}.ckpt".format(
                            epoch - self.num_models_to_keep
                        ),
                    )
                )
            except:
                print("not enough models there yet, nothing to delete")

        if optimizer is not None:
            torch.save(
                optimizer.state_dict(),
                os.path.join(self.opt.log_path, "optim_{}.ckpt".format(epoch)),
            )

            try:
                os.remove(
                    os.path.join(
                        self.opt.log_path,
                        "optim_{}.ckpt".format(epoch - self.num_models_to_keep),
                    )
                )
            except:
                print("not enough models there yet, nothing to delete")

        # Save hyper-parameters
        path = os.path.join(self.opt.log_path, "log.txt")
        with open(path, "w+") as cur_file:
            cur_file.write(str(self.opt))
            if accuracy is not None:
                cur_file.write("Top 1 -  accuracy: " + str(accuracy))
            if acc5 is not None:
                cur_file.write("Top 5 - Accuracy: " + str(acc5))
            if final_test and accuracy is not None:
                cur_file.write(" Very Final testing accuracy: " + str(accuracy))
            if final_test and acc5 is not None:
                cur_file.write(" Very Final testing top 5 - accuracy: " + str(acc5))

        # Save losses throughout training and plot
        self.np_save(
            os.path.join(self.opt.log_path, "train_loss"), np.array(self.train_loss, dtype=object)
        )

        if self.val_loss is not None:
            self.np_save(
                os.path.join(self.opt.log_path, "val_loss"), np.array(self.val_loss, dtype=object)
            )

        self.draw_loss_curve()

        if accuracy is not None:
            # self.np_save(os.path.join(self.opt.log_path, "accuracy"), accuracy)
            np.save(os.path.join(self.opt.log_path, "accuracy"), accuracy)



        if final_test:
            # self.np_save(os.path.join(self.opt.log_path, "final_accuracy"), accuracy)
            np.save(os.path.join(self.opt.log_path, "final_accuracy"), accuracy)
            # self.np_save(os.path.join(self.opt.log_path, "final_loss"), final_loss)
            np.save(os.path.join(self.opt.log_path, "final_loss"), final_loss)

    def draw_loss_curve(self):
        for idx, loss in enumerate(self.train_loss):
            lst_iter = np.arange(len(loss))
            plt.plot(lst_iter, np.array(loss), "-b", label="train loss")

            if (
                self.loss_last_training is not None
                and len(self.loss_last_training) > idx
            ):
                lst_iter = np.arange(len(self.loss_last_training[idx]))
                plt.plot(lst_iter, self.loss_last_training[idx], "-g")

            if self.val_loss is not None and len(self.val_loss) > idx:
                lst_iter = np.arange(len(self.val_loss[idx]))
                plt.plot(lst_iter, np.array(self.val_loss[idx]), "-r", label="val loss")

            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.legend(loc="upper right")
            # plt.axis([0, max(200,len(loss)+self.opt.encoder_config.start_epoch), 0, -round(np.log(1/(self.opt["negative_samples"]+1)),1)])

            # save image
            plt.savefig(os.path.join(self.opt.log_path, f"loss_{idx}.png"))
            try:
                tikzplotlib.save(os.path.join(self.opt.log_path, f"loss_{idx}.tex"))
            except:
                pass
            plt.close()

    def append_train_loss(self, train_loss):
        for idx, elem in enumerate(train_loss):
            self.train_loss[idx].append(elem)

    def append_val_loss(self, val_loss):
        for idx, elem in enumerate(val_loss):
            self.val_loss[idx].append(elem)
