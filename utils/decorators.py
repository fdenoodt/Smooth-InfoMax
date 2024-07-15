import time

from config_code.config_classes import OptionsConfig
from utils.utils import get_wandb_project_name, initialize_wandb
import wandb
import torch
import glob


def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time of {func.__name__}: {end_time - start_time} seconds")
        return result

    return wrapper


def wandb_decorator(func):
    def wrapper(options: OptionsConfig, *args, **kwargs):
        assert type(options) == OptionsConfig, \
            ("First argument must be an OptionsConfig object. "
             "When using this decorator, the first argument must be the options object."
             "eg: @wandb_decorator\n"
             "def _main(options: OptionsConfig):"
             "    ...")

        if options.use_wandb:
            project_name, run_name = get_wandb_project_name(options)
            initialize_wandb(options, project_name, run_name)

        result = func(options, *args, **kwargs)

        if options.use_wandb:
            wandb.finish()
        return result

    return wrapper
