"""
This file is used to get the options for the experiment. It is used to get the options from the config file and override.
Example usages:
    python main.py log_dir
    python main.py log_dir enc_gim_audio
    python main.py logs enc_gim_audio --overrides encoder_config.dataset.limit_train_batches=0.01 encoder_config.dataset.limit_validation_batches=0.01
    python main.py logs enc_gim_audio --overrides encoder_config.num_epochs=10 encoder_config.learning_rate=0.0001
    python main.py logs enc_gim_audio --overrides encoder_config.num_epochs=10 classifier_config_phones.num_epochs=10 classifier_config_phones.learning_rate=0.0001

config_file and overrides are optional. If config_file is not provided, it will use the default config file.
"""

import argparse

from config_code.config_classes import Dataset, DecoderLoss

from configs.enc_default import _get_options as default_get_options

# Create the parser
parser = argparse.ArgumentParser(description='Process some integers.')

# Add the arguments
parser.add_argument('experiment_name', type=str, help='The experiment name')
parser.add_argument('config_file', type=str, nargs='?', default=None, help='The config file name')
parser.add_argument('--overrides', nargs='*', help='The overrides for the config parameters')

# Parse the arguments
args = parser.parse_args()

experiment_name = args.experiment_name

if args.config_file:
    # eg: `from configs.enc_gim_audio import get_options`
    exec(f'from configs.{args.config_file} import _get_options')
else:
    _get_options = default_get_options

# Get the options
assert callable(_get_options), f"_get_options is not callable: {_get_options}"
_options = _get_options(experiment_name=experiment_name)

# Override the parameters if they are provided
if args.overrides is not None:
    for override in args.overrides:
        key, value = override.split('=')

        # Handle enums (Dataset Enum)
        if key.endswith('dataset.dataset'):  # convert into Dataset enum
            # assert number
            assert value.isdigit(), f"Value for {key} should be an integer, but it is {value}"
            value = Dataset(int(value))

        # Loss enum of decoder
        if key.endswith('decoder_config.decoder_loss'):
            assert value.isdigit(), f"Value for {key} should be an integer, but it is {value}"
            value = DecoderLoss(int(value))

        keys = key.split('.')
        last_key = keys.pop()
        obj = _options
        for k in keys:
            if not hasattr(obj, k):
                raise AttributeError(f"Object {obj} does not have attribute {k}")
            obj = getattr(obj, k)

        setattr(obj, last_key, type(getattr(obj, last_key))(value))

get_options = lambda: _options
