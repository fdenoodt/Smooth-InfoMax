from arg_parser import arg_parser
from config_code.config_classes import ModelType, OptionsConfig, ClassifierConfig, Dataset, Label
from decoder.my_data_module import MyDataModule
from linear_classifiers.downstream_classification import ClassifierModel
from models.load_audio_model import load_classifier
from options import get_options
from utils import logger
from utils.decorators import init_decorator, wandb_resume_decorator, timer_decorator
from utils.utils import get_classif_log_path


@init_decorator  # sets seed and clears cache etc
@wandb_resume_decorator
@timer_decorator
def main(opt: OptionsConfig, classifier_config: ClassifierConfig):
    # loads the log path where classifier is stored
    arg_parser.create_log_path(opt, add_path_var=get_classif_log_path(opt, classifier_config))

    classifier = ClassifierModel(opt, classifier_config)
    data_module = MyDataModule(opt.post_hoc_dataset)

    # load the classifier model
    classifier.classifier = load_classifier(opt, classifier.classifier)  # update Lightning module as well!!!


if __name__ == "__main__":
    options: OptionsConfig = get_options()
    c_config: ClassifierConfig = options.classifier_config
    options.model_type = ModelType.ONLY_DOWNSTREAM_TASK
    main(options, c_config)
