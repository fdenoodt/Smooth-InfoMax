# Variational Greedy InfoMax (V-GIM)

This repository contains the code of my thesis on interpretable representation learning with latent space constraints.
The code for both V-GIM and the decoder is included.

## Abstract

We introduces Variational Greedy InfoMax (V-GIM), a self-supervised representation
learning method that incorporates an interpretability constraint into both the network and the
loss function. V-GIM’s architecture is split up into modules, each individually optimised to
promote interpretability by imposing constraints on their respective latent spaces. This approach
enables the analysis of the underlying structures in the internal and final representations.
Inspired by Variational Autoencoders (VAE), V-GIM represents each module’s representations
as samples from Gaussian distributions. These representations are optimised to maximise
the mutual information between temporally nearby patches using the InfoNCE bound introduced
in (van den Oord et al., 2019). However, V-GIM introduces a regularisation term to the loss
function, encouraging distributions to approximate the standard normal distribution, thereby
constraining the latent space of each module. By enforcing these latent space constraints, VGIM
ensures that sampling a random point from the standard normal distribution within a
specific latent space is likely to correspond to a meaningful data point in the original space,
similar to the points in the dataset. This constraint also promotes smooth transitions in the
latent space with respect to the mutual information. V-GIM’s latent space is utilised to train a
decoder, enabling the analysis of the underlying structure of representations at different levels in
the architecture while ensuring good generalisation. Additionally, we argue that V-GIM’s latent
space constraint is related to theories from VAEs leading to disentangled representations, which
could potentially enable easier analysis of the model through post-hoc explainability approaches.
We evaluate V-GIM’s performance on sequential speech data. V-GIM achieves similar performance
to its less interpretable counterpart, GIM, and even outperforms it in deeper modules
due to V-GIM’s internal batch normalisation mechanism. We examine the potential of V-GIM’s
representation variance as a data augmentation tool for downstream tasks with limited labelled
data and demonstrate that the variability in representations does not contribute to improved
performance. Finally, we provide insights into V-GIM’s internal representations. We demonstrate
that the representations learn to differentiate between vowel information while consonant
information appears to be less prominent. Additionally, we discuss which internal neurons are
sensitive to which vowels, further demonstrating the interpretability of V-GIM.



<img src="assets\image-20230613110122953.png" alt="image-20230613110122953" style="zoom: 33%;" />

<img src="./assets/image-20230613111315897.png" alt="image-20230613111315897" style="zoom:33%;" />





<img src="./assets/image-20230613110900073.png" alt="image-20230613110900073" style="zoom: 67%;" />

## Installation

```bash
git clone https://github.com/oBoii/Smooth-InfoMax
./download_audio_data.sh
./audio_traineval.sh
```

```bash
cd /project_ghent/Smooth-InfoMax/ && git fetch && git pull && chmod +x ./audio_traineval.sh && ./audio_traineval.sh
```

```undefined
watch -n 0.5 nvidia-smi
```

## Running the project

De boer encoder + classifier:

```shell
bash -c "set -e;
outp_dir='bart_audio_distribs_distr=false_kld=0';
config_file='sim_audio_distr_false';
override='encoder_config.kld_weight=0 encoder_config.dataset.dataset=5 encoder_config.dataset.batch_size=128';


cd /project_antwerp/Smooth-InfoMax/;
git fetch; git pull;
pip install soundfile;
pip install librosa;
pip install tikzplotlib;

echo 'Training the Greedy InfoMax Model on audio data (librispeech)'; 
python -m main $outp_dir $config_file --overrides $override encoder_config.dataset.limit_train_batches=0.01 encoder_config.dataset.limit_validation_batches=0.01;

echo 'Testing syllable classification'; 
python -m linear_classifiers.logistic_regression_syllables $outp_dir $config_file --overrides $override;"
```

GIM:
```shell
outp_dir='bart_audio_distribs_distr=false_kld=0';
config_file='sim_audio_distr_false';
override='encoder_config.kld_weight=0 encoder_config.dataset.dataset=5 encoder_config.dataset.batch_size=128';
```

SIM: (kld=0.0033)
```shell
outp_dir='bart_audio_distribs_distr=true_kld=0.0033';
config_file='sim_audio_distr_true';
override='encoder_config.kld_weight=0.0033 encoder_config.dataset.dataset=5 encoder_config.dataset.batch_size=128';
```

CPC
```shell
outp_dir='cpc';
config_file='sim_audio_cpc';
override='encoder_config.dataset.dataset=5 encoder_config.dataset.batch_size=128';
```

```shell
python -m decoder.train_decoder ${output_dir} ${config_file} bart_audio_distribs_distr=true_kld=0.0033 sim_audio_distr_true --overrides encoder_config.kld_weight=0.0033 encoder_config.dataset.dataset=5 encoder_config.dataset.batch_size=128 decoder_config.decoder_loss=2
```


