from options import get_options
from encoder.train import run_configuration

if __name__ == "__main__":
    options = get_options(experiment_name='temp')
    run_configuration(options)
