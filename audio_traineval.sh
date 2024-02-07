echo "Training the Greedy InfoMax Model on audio data (librispeech)"
python -m main.py

echo "Testing the Greedy InfoMax Model for phone classification"
python -m linear_classifiers.logistic_regression_phones

echo "Testing the Greedy InfoMax Model for speaker classification"
python -m linear_classifiers.logistic_regression_speaker