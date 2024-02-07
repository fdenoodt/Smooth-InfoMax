echo "Downloading audio datasets:"
cd datasets || exit
wget http://www.openslr.org/resources/12/train-clean-100.tar.gz
tar -xzf train-clean-100.tar.gz || exit
mkdir LibriSpeech100_labels_split
cd LibriSpeech100_labels_split || exit
wget -O test_split.txt https://drive.google.com/uc?id=1vSHmncPsRY7VWWAd_BtoWs9-fQ5cBrEB
wget -O train_split.txt https://drive.google.com/uc?id=1ubREoLQu47_ZDn39YWv1wvVPe2ZlIZAb
wget -O converted_aligned_phones.zip https://drive.google.com/uc?id=1bLuDkapGBERG_VYPS7fNZl5GXsQ9z3p2
unzip converted_aligned_phones.zip

# For the subset of the dataset
cd .. # datasets
mkdir LibriSpeech100_labels_split_subset
cd LibriSpeech100_labels_split_subset || exit
wget -O test_split.txt https://drive.usercontent.google.com/u/3/uc?id=1rhSFg7DEDGP4ndZhT8bd5IfIHcHfE60D
wget -O train_split.txt https://drive.usercontent.google.com/u/3/uc?id=18z2XTshJSiuxXnxjmUoAZQGnkezxOSV7
wget -O converted_aligned_phones.txt https://drive.usercontent.google.com/u/3/uc?id=1igjFaJJpwoHLJzZMdFvv9wRc167NIFcK

cd ../..