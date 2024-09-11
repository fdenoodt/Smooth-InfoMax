# Smooth InfoMax

This repository contains the code for the paper Smooth InfoMax -- Towards Easier Post-Hoc
Interpretability. [[2408.12936] Smooth InfoMax--Towards easier Post-Hoc interpretability (arxiv.org)](https://arxiv.org/abs/2408.12936)



<img src="./assets/image-20230613111315897.png" alt="image-20230613111315897" style="zoom:33%;" />

## Abstract

We introduce Smooth InfoMax (SIM), a novel method for self-supervised representation learning that incorporates an
interpretability constraint into the learned representations at various depths of the neural network. SIM's architecture
is split up into probabilistic modules, each locally optimized using the InfoNCE bound. Inspired by VAEs, the
representations from these modules are designed to be samples from Gaussian distributions and are further constrained to
be close to the standard normal distribution. This results in a smooth and predictable space, enabling traversal of the
latent space through a decoder for easier post-hoc analysis of the learned representations. We evaluate SIM's
performance on sequential speech data, showing that it performs competitively with its less interpretable counterpart,
Greedy InfoMax (GIM). Moreover, we provide insights into SIM's internal representations, demonstrating that the
contained information is less entangled throughout the representation and more concentrated in a smaller subset of the
dimensions. This further highlights the improved interpretability of SIM.



<img src="assets\image-20230613110122953.png" alt="image-20230613110122953" style="zoom: 33%;" />

## Running the code and reproducing the experiments

### LibriSpeech dataset

- <details>
    <summary>Download LibriSpeech</summary>

    ```shell
        mkdir datasets
        cd datasets || exit
        wget http://www.openslr.org/resources/12/train-clean-100.tar.gz
        tar -xzf train-clean-100.tar.gz || exit
        mkdir LibriSpeech100_labels_split
        cd LibriSpeech100_labels_split || exit
        gdown https://drive.google.com/uc?id=1vSHmncPsRY7VWWAd_BtoWs9-fQ5cBrEB # test split
        gdown https://drive.google.com/uc?id=1ubREoLQu47_ZDn39YWv1wvVPe2ZlIZAb # train split
        gdown https://drive.google.com/uc?id=1bLuDkapGBERG_VYPS7fNZl5GXsQ9z3p2 # converted_aligned_phones.zip
        unzip converted_aligned_phones.zip
        cd ../..
    ```
  </details>

- <details>
    <summary>Set required variables</summary>

    ```shell
    # Required setup (overwrites hyperparameters and sets up wandb for logging)
    
    # By default these args will run SIM. If you want to run GIM instead, change `sim_audio_de_boer_distr_true` to `sim_audio_de_boer_distr_false`
    override='./logs sim_audio_de_boer_distr_true \
    --overrides encoder_config.dataset.num_workers=4 speakers_classifier_config.dataset.num_workers=4 phones_classifier_config.dataset.num_workers=4 decoder_config.dataset.num_workers=4 \
    encoder_config.dataset.dataset=1 phones_classifier_config.dataset.dataset=1 speakers_classifier_config.dataset.dataset=1 decoder_config.dataset.dataset=1 \
    encoder_config.dataset.batch_size=8 decoder_config.dataset.batch_size=8 speakers_classifier_config.dataset.batch_size=8 phones_classifier_config.dataset.batch_size=8 \
    seed=4 encoder_config.num_epochs=100 speakers_classifier_config.encoder_num=99 phones_classifier_config.encoder_num=99 decoder_config.encoder_num=99 decoder_config.num_epochs=50 \
    phones_classifier_config.num_epochs=10 speakers_classifier_config.num_epochs=10 speakers_classifier_config.gradient_clipping=1.0 phones_classifier_config.gradient_clipping=1.0 \
    encoder_config.kld_weight=0.001 \
    wandb_project_name=TODO wandb_entity=TODO '; # Please update this line in
    
    # Log into WandB
    wandb login XXXXX-WANDB-KEY-PLEASE-USE-YOUR-OWN-XXXX;
    ```
  </details>

- <details>
    <summary>Run SIM or GIM</summary>

    ```shell
    python -m main $override;
    ```
    </details>

- <details>
    <summary>Run the classifiers</summary>

    ```shell
    echo 'Training classifier - speakers'; 
    python -m linear_classifiers.logistic_regression_speaker $override \
        speakers_classifier_config.dataset.dataset=1 \
        speakers_classifier_config.bias=True \
        encoder_config.deterministic=True;
    
    echo 'Training classifier - phones'; 
    python -m linear_classifiers.logistic_regression_phones $override \
        phones_classifier_config.dataset.dataset=1 \
        speakers_classifier_config.bias=True \
        encoder_config.deterministic=True;
  ```
  </details>

- <details>
    <summary>Run the interpretability analysis</summary>

    ```shell
    # Perform speaker classification with different encoder settings
    python -m linear_classifiers.logistic_regression_speaker $override \
        encoder_config.deterministic=False \
        speakers_classifier_config.bias=False \
        speakers_classifier_config.encoder_module=0 \
        speakers_classifier_config.encoder_layer=-1;
    
    python -m post_hoc_analysis.interpretability.main_speakers_analysis $override \
        encoder_config.deterministic=False \
        speakers_classifier_config.bias=False \
        speakers_classifier_config.encoder_module=0 \
        speakers_classifier_config.encoder_layer=-1;
    
    python -m linear_classifiers.logistic_regression_speaker $override \
        encoder_config.deterministic=False \
        speakers_classifier_config.bias=False \
        speakers_classifier_config.encoder_module=1 \
        speakers_classifier_config.encoder_layer=-1;
    
    python -m post_hoc_analysis.interpretability.main_speakers_analysis $override \
        encoder_config.deterministic=False \
        speakers_classifier_config.bias=False \
        speakers_classifier_config.encoder_module=1 \
        speakers_classifier_config.encoder_layer=-1;
    
    python -m linear_classifiers.logistic_regression_speaker $override \
        encoder_config.deterministic=False \
        speakers_classifier_config.bias=False \
        speakers_classifier_config.encoder_module=2 \
        speakers_classifier_config.encoder_layer=-1;
    
    python -m post_hoc_analysis.interpretability.main_speakers_analysis $override \
        encoder_config.deterministic=False \
        speakers_classifier_config.bias=False \
        speakers_classifier_config.encoder_module=2 \
        speakers_classifier_config.encoder_layer=-1;
    
    # Train decoder with different encoder modules
    python -m decoder.train_decoder $override \
        decoder_config.decoder_loss=0 \
        decoder_config.dataset.dataset=1 \
        decoder_config.encoder_module=0 \
        decoder_config.encoder_layer=-1;
    
    python -m decoder.train_decoder $override \
        decoder_config.decoder_loss=0 \
        decoder_config.dataset.dataset=1 \
        decoder_config.encoder_module=1 \
        decoder_config.encoder_layer=-1;
    
    python -m decoder.train_decoder $override \
        decoder_config.decoder_loss=0 \
        decoder_config.dataset.dataset=1 \
        decoder_config.encoder_module=2 \
        decoder_config.encoder_layer=-1;
    ```
  </details>

### Artificial Speech dataset

- <details>
  <summary>Download the dataset</summary>

   ```shell
    git clone https://github.com/fdenoodt/Artificial-Speech-Dataset
    cp -r Artificial-Speech-Dataset/* datasets/corpus/
   ```
  </details>

- <details>
    <summary>Set required variables</summary>

    ```shell
    # Required setup (overwrites hyperparameters and sets up wandb for logging)
      
    # By default these args will run SIM. If you want to run GIM instead, change `sim_audio_de_boer_distr_true` to `sim_audio_de_boer_distr_false`
    override='./logs sim_audio_de_boer_distr_true \
    --overrides \
    encoder_config.dataset.num_workers=4 \
    syllables_classifier_config.dataset.num_workers=4 \
    decoder_config.dataset.num_workers=4 \
    encoder_config.use_batch_norm=False \
    use_wandb=True \
    wandb_project_name=TODO wandb_entity=TODO '; # Please update this line
      
    # Log into WandB
    wandb login XXXXX-WANDB-KEY-PLEASE-USE-YOUR-OWN-XXXX;
    ```
  </details>

- <details>
    <summary>Run SIM or GIM</summary>

    ```shell
    python -m main $override;
    ```
    </details>

- <details>
    <summary>Run the classifiers</summary>

    ```shell
    echo 'Training classifier - syllables'; 
    python -m linear_classifiers.logistic_regression $override \
        syllables_classifier_config.bias=True \
        syllables_classifier_config.dataset.labels=syllables \
        encoder_config.deterministic=True;
    
    echo 'Training classifier - vowels'; 
    python -m linear_classifiers.logistic_regression $override \
        syllables_classifier_config.bias=True \
        syllables_classifier_config.dataset.labels=vowels \
        encoder_config.deterministic=True;
    ```
  </details>

- <details>
    <summary>Run the interpretability analysis</summary>

    ```shell
    # Perform vowel classification with different encoder settings
    python -m linear_classifiers.logistic_regression $override \
        encoder_config.deterministic=False \
        syllables_classifier_config.bias=False \
        syllables_classifier_config.dataset.labels=vowels \
        syllables_classifier_config.encoder_module=0 \
        syllables_classifier_config.encoder_layer=-1;
    
    echo 'vowel weight plots'; 
    python -m post_hoc_analysis.interpretability.main_vowel_classifier_analysis $override \
        encoder_config.deterministic=False \
        syllables_classifier_config.bias=False \
        syllables_classifier_config.dataset.labels=vowels \
        syllables_classifier_config.encoder_module=0 \
        syllables_classifier_config.encoder_layer=-1;
    
    python -m linear_classifiers.logistic_regression $override \
        encoder_config.deterministic=False \
        syllables_classifier_config.bias=False \
        syllables_classifier_config.dataset.labels=vowels \
        syllables_classifier_config.encoder_module=1 \
        syllables_classifier_config.encoder_layer=-1;
    
    echo 'vowel weight plots'; 
    python -m post_hoc_analysis.interpretability.main_vowel_classifier_analysis $override \
        encoder_config.deterministic=False \
        syllables_classifier_config.bias=False \
        syllables_classifier_config.dataset.labels=vowels \
        syllables_classifier_config.encoder_module=1 \
        syllables_classifier_config.encoder_layer=-1;
    
    python -m linear_classifiers.logistic_regression $override \
        encoder_config.deterministic=False \
        syllables_classifier_config.bias=False \
        syllables_classifier_config.dataset.labels=vowels \
        syllables_classifier_config.encoder_module=2 \
        syllables_classifier_config.encoder_layer=-1;
    
    echo 'vowel weight plots'; 
    python -m post_hoc_analysis.interpretability.main_vowel_classifier_analysis $override \
        encoder_config.deterministic=False \
        syllables_classifier_config.bias=False \
        syllables_classifier_config.dataset.labels=vowels \
        syllables_classifier_config.encoder_module=2 \
        syllables_classifier_config.encoder_layer=-1;
    
    # Train decoder with different encoder modules
    python -m decoder.train_decoder $override \
        decoder_config.decoder_loss=0 \
        decoder_config.dataset.dataset=4 \
        decoder_config.encoder_module=0 \
        decoder_config.encoder_layer=-1;
    
    python -m decoder.train_decoder $override \
        decoder_config.decoder_loss=0 \
        decoder_config.dataset.dataset=4 \
        decoder_config.encoder_module=1 \
        decoder_config.encoder_layer=-1;
  
    python -m decoder.train_decoder $override \
        decoder_config.decoder_loss=0 \
        decoder_config.dataset.dataset=4 \
        decoder_config.encoder_module=2 \
        decoder_config.encoder_layer=-1;
    ```
    </details>