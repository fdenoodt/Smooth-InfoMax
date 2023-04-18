import torch
import numpy as np
from options_classify_syllables import get_options

import main_classify_syllables as main


if __name__ == "__main__":
    # random seeds
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)

    ENC_LOGS_ROOT = r"E:\thesis_logs\logs"
    encoder_paths = [fr"{ENC_LOGS_ROOT}\de_boer_TWO_MODULE_V3_dim32_kld_weight=0\model_790.ckpt",
                      fr"{ENC_LOGS_ROOT}\de_boer_TWO_MODULE_V3_dim32_kld_weight=0.0033\model_790.ckpt"]
    
    # r"D:\thesis_logs\logs\de_boer_TWO_MODULE_V3_dim32_kld_weight=0.0033 !!/model_1599.ckpt"]
    encoder_names = ["kld=0", "kld=0.0033"]
    for subset_size in [ "2", "4", "8", "16", "32", "64", "128", "all"]:
        for encoder_path, encoder_name in zip(encoder_paths, encoder_names):
            print("")
            print("")
            print(f"*********Trying {encoder_name} subset={subset_size}*********")
            OPTIONS = get_options()
            OPTIONS['subset'] = subset_size
            OPTIONS['cpc_model_path'] = encoder_path

            main.run_configuration(
                OPTIONS, f"linear_model_{encoder_name}_subset={subset_size}_epoch=790")
