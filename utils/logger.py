import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import copy

from linear_classifiers.downstream_classification import ClassifierModel

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
            try:  # just for the decoder
                np.savetxt(f"{path}_{idx}.csv", item, delimiter=",")
            except:
                pass

    def create_log(
            self,
            model,
            epoch=0,
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

                print("not enough models there yet, nothing to delete")

    def create_classifier_log(self, classifier: ClassifierModel, epoch=0):
        """
        classifier contains both the encoder as the linear classifier
        """
        assert isinstance(classifier, ClassifierModel), "classifier is not a ClassifierModel"

        print("Saving model and log-file to " + self.opt.log_path)

        # Save the model checkpoint
        torch.save(
            classifier.state_dict(),
            os.path.join(self.opt.log_path, "model_{}.ckpt".format(epoch)),
        )

    def create_decoder_log(self, decoder, epoch):
        print("Saving model and log-file to " + self.opt.log_path)

        # Save the model checkpoint
        torch.save(
            decoder.state_dict(),
            os.path.join(self.opt.log_path, "decoder_{}.ckpt".format(epoch)),
        )
