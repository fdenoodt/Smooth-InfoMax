from options import get_options
from main import run_configuration

if __name__ == "__main__":
  EXPERIMENT_NAME = 'temp'

  decay_rates = [0.02, 0.03]
  # decay_rates = [0.02, 0.03, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]

  for decay_rate in decay_rates:
      print("hi")
      OPTIONS = get_options(f"{EXPERIMENT_NAME}_decay_rate_{decay_rate}")
      OPTIONS['decay_rate'] = decay_rate

      run_configuration(OPTIONS)
