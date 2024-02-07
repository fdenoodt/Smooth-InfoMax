pip install soundfile
pip install librosa
pip install tikzplotlib

echo "Training the Greedy InfoMax Model on audio data (librispeech)"
python -m main temp enc_gim_audio # `enc_gim_audio` is the config file (in configs/ directory) for Greedy InfoMax Model

echo "Testing the Greedy InfoMax Model for phone classification"
python -m linear_classifiers.logistic_regression_phones temp enc_gim_audio

echo "Testing the Greedy InfoMax Model for speaker classification"
python -m linear_classifiers.logistic_regression_speaker temp enc_gim_audio