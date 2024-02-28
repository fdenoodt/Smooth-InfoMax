# equivalent of sim_audio_distr_false (which was GIM), but now with the CPC model, so single module
import torch
from config_code.config_classes import OptionsConfig
from config_code.cpc_setup import CPCSetup


def _get_options(experiment_name) -> OptionsConfig:
    sim_setup = CPCSetup()
    options = sim_setup.get_options(experiment_name)

    return options


if __name__ == '__main__':
    print(f"Cuda is available: {torch.cuda.is_available()}")
