# python -m linear_classifiers.logistic_regression_syllables temp sim_audio_distr_true --overrides syllables_classifier_config.encoder_num=9
from linear_classifiers.logistic_regression import main

if __name__ == "__main__":
    wandb, wandb_is_on, _ = main(syllables=True)  # syllables classification
    if wandb_is_on:
        wandb.finish()
