from options import get_options
from main import run_configuration

if __name__ == "__main__":
  EXPERIMENT_NAME = 'libri_cpc_kld_weight'

  kld_weights = [0, 0.32]
  # kld_weights = [0.031, 0.032, 0.033, 0.034, 0.035, 0.036, 0.037, 0.038, 0.039]
  # .031, 0.032 appeared the best

  for kld_weight in kld_weights:
      try:
        OPTIONS = get_options(f"{EXPERIMENT_NAME}_kld_weight={kld_weight}")
        OPTIONS['kld_weight'] = kld_weight

        run_configuration(OPTIONS)
      except Exception as e:
        print("********************************************")
        print(f"Error: {e}, {kld_weight}")
        print("********************************************")