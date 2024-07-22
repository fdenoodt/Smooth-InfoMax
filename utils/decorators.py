import time

from arg_parser import arg_parser
from config_code.config_classes import OptionsConfig
from utils.utils import get_wandb_project_name, initialize_wandb, retrieve_existing_wandb_run_id
import wandb
import torch
import glob
import gc
from utils.utils import set_seed


def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time of {func.__name__}: {end_time - start_time} seconds")
        return result

    return wrapper


def wandb_decorator(func):
    """
    Initialize wandb if options.use_wandb is True, and finish the run after the function is done.
    Warning: ensure that the first argument of the function with this decorator is the options object!
    """

    def wrapper(options: OptionsConfig, *args, **kwargs):
        assert type(options) == OptionsConfig, \
            ("First argument must be an OptionsConfig object. "
             "When using this decorator, the first argument must be the options object."
             "eg: @wandb_decorator\n"
             "def _main(options: OptionsConfig):"
             "    ...")

        if options.use_wandb:
            entity, project_name, run_name = get_wandb_project_name(options)
            initialize_wandb(options, entity, project_name, run_name)

        result = func(options, *args, **kwargs)

        if options.use_wandb:
            wandb.finish()
        return result

    return wrapper


def wandb_resume_decorator(func):
    def wrapper(options: OptionsConfig, *args, **kwargs):
        assert type(options) == OptionsConfig, \
            ("See `wandb_decorator` for more information.")

        if options.use_wandb:
            run_id, project_name = retrieve_existing_wandb_run_id(options)
            wandb.init(id=run_id, resume="allow", project=project_name, entity=options.wandb_entity)

        result = func(options, *args, **kwargs)

        if options.use_wandb:
            wandb.finish()

        return result

    return wrapper


def init_decorator(func):
    def init(options: OptionsConfig):
        torch.set_float32_matmul_precision('medium')
        torch.cuda.empty_cache()
        gc.collect()

        # set random seeds
        set_seed(options.seed)

        arg_parser.create_log_path(options)

    def wrapper(options: OptionsConfig, *args, **kwargs):
        assert type(options) == OptionsConfig, \
            ("See `wandb_decorator` for more information.")

        init(options)
        result = func(options, *args, **kwargs)

        return result

    return wrapper
