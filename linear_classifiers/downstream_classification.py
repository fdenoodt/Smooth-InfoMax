"""
This script is only for syllable/vowel classification. For speaker/phoeme classification, see other scripts.
"""

# example python call:
# python -m linear_classifiers.logistic_regression_syllables  final_bart/bart_full_audio_distribs_distr=true_kld=0 sim_audio_distr_false
# or
# python -m linear_classifiers.logistic_regression_syllables temp sim_audio_de_boer_distr_true --overrides encoder_config.kld_weight=0.01 encoder_config.num_epochs=2 syllables_classifier_config.encoder_num=1 syllables_classifier_config.num_epochs=3 use_wandb=False train=True
from typing import Optional

import lightning
import torch
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger

from arg_parser import arg_parser
from config_code.config_classes import OptionsConfig, ModelType, Dataset, ClassifierConfig, Label
from decoder.my_data_module import MyDataModule
from models import load_audio_model
from models.full_model import FullModel
from models.load_audio_model import load_classifier
from models.loss_supervised_syllables import Syllables_Loss
from options import get_options
from utils import logger
from utils.decorators import timer_decorator, wandb_resume_decorator, init_decorator
from utils.utils import get_nb_classes, \
    get_classif_log_path, get_wandb_audio_classific_key


class ClassifierModel(lightning.LightningModule):
    def __init__(self, options: OptionsConfig, classifier_config: ClassifierConfig):
        super(ClassifierModel, self).__init__()
        self.options = options
        self.classifier_config = classifier_config
        self.encoder, _ = load_audio_model.load_model_and_optimizer(
            options,
            classifier_config,
            reload_model=True,  # if opt.model_type == ModelType.ONLY_DOWNSTREAM_TASK else False,
            calc_accuracy=True,
            num_GPU=1,
        )
        self.encoder: FullModel = self.encoder  # for type hinting
        self.classifier: Syllables_Loss = self.setup_classifier(options, classifier_config)

    def setup_classifier(self, opt: OptionsConfig, classifier_config: ClassifierConfig):
        """
        WARNING: bias = False is only used for vowel classifier on the ConvLayer. It's not supported beyond that (eg regression layer).
        It is only used for the latent space analysis, not used for performance evaluation.

        - 256 is the output of the regression layer (conventional case).
        - 512 is the output of the ConvLayer (only space analysis).
        """

        regr_hidden_dim = opt.encoder_config.architecture.modules[0].regressor_hidden_dim
        cnn_hidden_dim = opt.encoder_config.architecture.modules[0].cnn_hidden_dim
        bias = classifier_config.bias
        if bias:
            n_features = regr_hidden_dim
        else:
            n_features = cnn_hidden_dim

        num_classes = get_nb_classes(opt.get_classifier_key())

        # The loss class also contains the classifier!
        loss: Syllables_Loss = Syllables_Loss(opt, n_features, calc_accuracy=True, num_syllables=num_classes, bias=bias)
        return loss

    def configure_optimizers(self):
        if self.options.model_type == ModelType.FULLY_SUPERVISED:
            self.encoder.train()
            params = list(self.encoder.parameters()) + list(self.classifier.parameters())
        elif options.model_type == ModelType.ONLY_DOWNSTREAM_TASK:
            self.encoder.eval()
            params = list(self.classifier.parameters())
        else:
            raise ValueError(
                "Model type not supported for training classifier. "
                "(only FULLY_SUPERVISED or ONLY_DOWNSTREAM_TASK)")

        learning_rate = self.classifier_config.learning_rate
        optimizer = torch.optim.Adam(params, lr=learning_rate)
        return optimizer

    def _get_representation(self, opt: OptionsConfig, method: callable,
                            # arg1 is mandatory, arg2 and arg3 are optional depending on the method
                            arg1, arg2: Optional[int], arg3: Optional[int]):
        def _forward(method, arg1, arg2: int, arg3: int):
            if isinstance(arg3, int):  # 3 args
                assert isinstance(arg2, int)
                z = method(arg1, arg2, arg3)
            elif isinstance(arg2, int):  # 2 args
                z = method(arg1, arg2)
            else:  # single arg
                z = method(arg1)
            return z

        if opt.model_type == ModelType.ONLY_DOWNSTREAM_TASK:
            with torch.no_grad():
                z = _forward(method, arg1, arg2, arg3)
            z = z.detach()
        else:  # opt.model_type == ModelType.FULLY_SUPERVISED
            z = _forward(method, arg1, arg2, arg3)
        return z

    def get_z(self, opt, context_model, model_input, regression: bool, which_module: int, which_layer: int):
        # Set regression=True, which_module=-1, which_layer=-1, for the conventional case (performance measurements).
        # `which_module` and `which_layer` are only used for latent space analysis of intermediate layers/modules.
        # For GIM/SIM, can specify module index. The layer is typically always -1 (last one).
        # For CPC, there is a single module, but a layer idx can be specified.

        if regression:
            assert which_module == -1 and which_layer == -1, "Regression layer doesn't have modules"

        if regression:  # typical case, includes regression layer
            method = context_model.module.forward_through_all_modules
            return self._get_representation(opt, method, model_input, None, None)

        # Conv module only used for latent space/interpretability analysis
        if which_module == -1 and which_layer == -1:
            method = context_model.module.forward_through_all_cnn_modules
            z = self._get_representation(opt, method, model_input, None, None)
        elif which_module >= 0 and which_layer == -1:
            method = context_model.module.forward_through_module  # takes 2 args (input, module)
            z = self._get_representation(opt, method, model_input, which_module, None)

        elif which_module >= 0 and which_layer >= 0:  # specific layer in specific module (for CPC for example)
            method = context_model.module.forward_through_layer  # takes 3 args (input, module, layer)
            z = self._get_representation(opt, method, model_input, which_module, which_layer)
        else:
            raise ValueError("Invalid layer/module specification")

        return z.permute(0, 2, 1)

    def forward(self, batch):
        # return self.encoder(x)
        (x, label) = batch
        z = self.get_z(self.options, self.encoder, x,
                       regression=self.classifier_config.bias,
                       which_module=self.classifier_config.encoder_module,
                       which_layer=self.classifier_config.encoder_layer)

        # mode_accuracy only useful when predicting from a single frame
        total_loss, accuracy, mode_accuracy = self.classifier.get_loss(x, z, z, label)
        return total_loss, accuracy, mode_accuracy

    def training_step(self, batch, batch_idx):
        loss, accuracies, mode_accuracy = self.forward(batch)

        wandb_section = get_wandb_audio_classific_key(self.options, self.classifier_config)
        self.log(f"{wandb_section}/Loss classification", loss, batch_size=self.options.post_hoc_dataset.batch_size)
        self.log(f"{wandb_section}/Train accuracy", accuracies, batch_size=self.options.post_hoc_dataset.batch_size)
        self.log(f"{wandb_section}/Mode accuracy", mode_accuracy, batch_size=self.options.post_hoc_dataset.batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        loss, accuracy, mode_accuracy = self.forward(batch)

        wandb_section = get_wandb_audio_classific_key(self.options, self.classifier_config)
        self.log(f"{wandb_section}/Test loss", loss, batch_size=self.options.post_hoc_dataset.batch_size)
        self.log(f"{wandb_section}/Test accuracy", accuracy, batch_size=self.options.post_hoc_dataset.batch_size)
        self.log(f"{wandb_section}/Test mode accuracy", mode_accuracy,
                 batch_size=self.options.post_hoc_dataset.batch_size)
        return loss


@init_decorator  # sets seed and clears cache etc
@wandb_resume_decorator
@timer_decorator
def main(opt: OptionsConfig, classifier_config: ClassifierConfig):
    assert classifier_config is not None, "Classifier config is not set"
    assert opt.model_type in [ModelType.FULLY_SUPERVISED,
                              ModelType.ONLY_DOWNSTREAM_TASK], "Model type not supported"
    if opt.post_hoc_dataset.dataset == Dataset.DE_BOER:
        assert opt.post_hoc_dataset.labels in [Label.SYLLABLES, Label.VOWELS], "Labels not supported"
    assert (opt.post_hoc_dataset.dataset in
            [Dataset.DE_BOER, Dataset.LIBRISPEECH, Dataset.RADIO]), "Dataset not supported"

    arg_parser.create_log_path(opt, add_path_var=get_classif_log_path(opt, classifier_config))
    logs = logger.Logger(opt)  # Will be used to save the classifier model for instance

    classifier = ClassifierModel(opt, classifier_config)
    data_module = MyDataModule(opt.post_hoc_dataset)

    trainer = Trainer(
        max_epochs=classifier_config.num_epochs,
        limit_train_batches=options.post_hoc_dataset.limit_train_batches,
        limit_val_batches=options.post_hoc_dataset.limit_validation_batches,
        logger=WandbLogger() if opt.use_wandb else None,
        log_every_n_steps=10,
        profiler="pytorch" if opt.profile else None,
        fast_dev_run=opt.fast_dev_run,
        overfit_batches=opt.overfit_batches,
    )

    if opt.train:
        try:
            # Train the model
            trainer.fit(classifier, data_module)
            logs.create_log(classifier.classifier)
        except KeyboardInterrupt:
            print("Training interrupted, saving log files")

    # regardless of training, test the model by loading the final checkpoint
    classifier.classifier = load_classifier(opt, classifier.classifier)  # update Lightning module as well!!!
    trainer.test(classifier, data_module)  # Test the model

    # Only for De Boer dataset
    print(f"Finished training {options.post_hoc_dataset.labels} classifier")


if __name__ == "__main__":
    # IMPORTANT TO SET classifier_config.dataset.labels=[syllables|vowels], classifier_config.bias=[True|False] in the config file
    options: OptionsConfig = get_options()
    c_config: ClassifierConfig = options.classifier_config

    options.model_type = ModelType.FULLY_SUPERVISED  # ModelType.FULLY_SUPERVISED
    [print("*" * 50) for _ in range(3)]
    print(f"Classifier config: {c_config}")
    print(f"Model type: {options.model_type}")
    [print("*" * 50) for _ in range(3)]

    main(options, c_config)

    # Fully supervised with randomly initialized backbone --> says if architecture is any good
    # rnd init backbone, call train.py with train=False, then call downstream_classification.py with classifier_config.encoder_num=0
    # Fully supervised with frozen backbone (so ModelType.ONLY_DOWNSTREAM_TASK) --> says if representations from cpc are any good
    # For optimal performance: fine tuning the backbone with the downstream task: fully supervised but with backbone weights from cpc
