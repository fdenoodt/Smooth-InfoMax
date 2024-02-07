from options import get_options
from encoder.train import run_configuration

if __name__ == "__main__":
    options = get_options()
    run_configuration(options)
