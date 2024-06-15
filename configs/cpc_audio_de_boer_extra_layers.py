"""
Non-standard CPC configuration for the De Boer dataset.
- only used for experiments related to classifier density plots
- for performance comparison with the conventional CPC model, use cpc_audio_de_boer_conventional.py
"""

import os
import torch
from config_code.config_classes import OptionsConfig, Dataset
from config_code.sim_setup import SIMSetup


def _get_options(experiment_name) -> OptionsConfig:
    config_file = os.path.basename(__file__)  # eg cpc_audio_de_boer.py
    sim_setup = SIMSetup(predict_distributions=False, dataset=Dataset.DE_BOER, config_file=config_file, is_cpc=True,
                         conventional_cpc=False)
    options = sim_setup.get_options(experiment_name)

    return options


if __name__ == '__main__':
    print(f"Cuda is available: {torch.cuda.is_available()}")
