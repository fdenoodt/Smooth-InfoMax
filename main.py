from options import get_options
from encoder.train import run_configuration

if __name__ == "__main__":
    options = get_options()
    assert options.experiment == "audio"


    print("*" * 80)
    print(options)
    print("*" * 80)
    print()

    run_configuration(options)
