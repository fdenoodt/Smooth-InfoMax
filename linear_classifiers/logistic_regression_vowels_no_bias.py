# python -m linear_classifiers.logistic_regression_vowels temp sim_audio_distr_true --overrides syllables_classifier_config.encoder_num=9
# or temp sim_audio_de_boer_distr_true --overrides syllables_classifier_config.encoder_num=9
from linear_classifiers.logistic_regression import main

if __name__ == "__main__":
    wandb, wandb_is_on, params = main(syllables=False, bias=False)  # Syllables classification
    if wandb_is_on:
        # wandb.log({f"C Singl layer bias={bias} {label_type}/FINAL Test accuracy": accuracy,
        #            f"C Singl layer bias={bias} {label_type}/FINAL Test loss": loss_epoch})

        #wandb.log({f"LatSpace/t-SNE_latent_space_{gim_name}": [wandb.Image(f"{save_dir}/{file}")]})
        wandb.finish()
