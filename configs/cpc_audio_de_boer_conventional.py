"""
Configuration file for the CPC model on the De Boer dataset with conventional CPC.
- always use this eg for performance comparison with the conventional CPC model
- for density plots, use the extra layers configuration (cpc_audio_de_boer_extra_layers.py)
"""

import os
import torch
from config_code.config_classes import OptionsConfig, Dataset
from config_code.sim_setup import SIMSetup


def _get_options(experiment_name) -> OptionsConfig:
    config_file = os.path.basename(__file__)  # eg cpc_audio_de_boer.py
    sim_setup = SIMSetup(predict_distributions=False, config_file=config_file, is_cpc=True,
                         conventional_cpc=True)
    options = sim_setup.get_options(experiment_name)

    return options


if __name__ == '__main__':
    print(f"Cuda is available: {torch.cuda.is_available()}")
