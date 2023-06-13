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

    # only learn vowels, not consonants
    for which_module in ["2", "1"]:
        subset_size = "all"
        encoder_name = "kld=0.0035"
        encoder_path = fr"{ENC_LOGS_ROOT}\de_boer_TWO_MODULE_V3_dim32_kld_weight=0.0035\model_790.ckpt"
        print(f"*********Trying {encoder_name} subset={subset_size} module={which_module} subs={subset_size} *********")
        OPTIONS = get_options()
        OPTIONS['subset'] = subset_size
        OPTIONS['cpc_model_path'] = encoder_path
        OPTIONS['which_module'] = which_module
        for include_bias_term in [False, True]:
            OPTIONS['include_bias_term'] = include_bias_term

            # try:
            main.run_configuration(
                OPTIONS, f"ONLY_VOWELS_linear_model_{encoder_name}_subset={subset_size}_bias={include_bias_term}_epoch=790")
            # except Exception as e:
            #     # store to file
            #     with open("error_log.txt", "a") as f:
            #         f.write(f"*********Trying {encoder_name} subset={subset_size} module={which_module} *********\n")
            #         f.write(str(e))
            #         f.write("\n")



    # # lr param tuning:
    # for which_module in ["1", "2"]:
    #     for subset_size in ["all"]:
    #         encoder_name = "kld=0.0035"
    #         encoder_path = fr"{ENC_LOGS_ROOT}\de_boer_TWO_MODULE_V3_dim32_kld_weight=0.0035\model_790.ckpt"
    #         for lr in [0.001, 0.1, 0.0001]: # lr_0.0010000 seems best

    #             print("")
    #             print("")
    #             print(f"*********Trying {encoder_name} subset={subset_size} module={which_module} subs={subset_size} *********")
    #             OPTIONS = get_options()
    #             OPTIONS['subset'] = subset_size
    #             OPTIONS['cpc_model_path'] = encoder_path
    #             OPTIONS['which_module'] = which_module
    #             OPTIONS['learning_rate'] = lr

    #             if subset_size == "all":
    #                 OPTIONS['num_epochs'] = 100

    #             # try:
    #             main.run_configuration(
    #                 OPTIONS, f"no_pooling_linear_model_{encoder_name}_subset={subset_size}_epoch=790")
    #             # except Exception as e:
    #             #     # store to file
    #             #     with open("error_log.txt", "a") as f:
    #             #         f.write(f"*********Trying {encoder_name} subset={subset_size} module={which_module} *********\n")
    #             #         f.write(str(e))
    #             #         f.write("\n")



    # # ** try 512 dim models
    # # r"D:\thesis_logs\logs\de_boer_TWO_MODULE_V3_dim32_kld_weight=0.0033 !!/model_1599.ckpt"]
    # for which_module in ["1", "2"]:
    #     subset_size = "all"
    #     encoder_name = "kld=0.0035"
    #     encoder_path = r"E:\thesis_logs\logs\VGIM_kld_weight=0.0035_diff_k_decay_rate=1/model_790.ckpt"
    #     for pool in ["not", "max"]:
    #         print("")
    #         print("")
    #         print(f"*********Trying {encoder_name} subset={subset_size} module={which_module} subs={subset_size} *********")
    #         OPTIONS = get_options()
    #         OPTIONS['subset'] = subset_size
    #         OPTIONS['cpc_model_path'] = encoder_path
    #         OPTIONS['which_module'] = which_module
    #         OPTIONS['pooling'] = pool

    #         # try:
    #         main.run_configuration(
    #             OPTIONS, f"512dim_model_pool={pool}_linear_model_{encoder_name}_subset={subset_size}_epoch=790")
    #         # except Exception as e:
    #         #     # store to file
    #         #     with open("error_log.txt", "a") as f:
    #         #         f.write(f"*********Trying {encoder_name} subset={subset_size} module={which_module} *********\n")
    #         #         f.write(str(e))
    #         #         f.write("\n")




    ## ** subsets experiment
    # # r"D:\thesis_logs\logs\de_boer_TWO_MODULE_V3_dim32_kld_weight=0.0033 !!/model_1599.ckpt"]
    # for which_module in ["2", "1"]:
    #     # for which_module in ["1", "2"]:
    #     for subset_size in ["all"]:
    #         # for subset_size in ["4", "8", "16", "32", "64", "128"]:
    #         # for subset_size in ["all"]:
    #         encoder_names = ["kld=0.0035", "kld=0"]
    #         for encoder_path, encoder_name in zip(encoder_paths, encoder_names):
    #             print("")
    #             print("")
    #             print(f"*********Trying {encoder_name} subset={subset_size} module={which_module} subs={subset_size} *********")
    #             OPTIONS = get_options()
    #             OPTIONS['subset'] = subset_size
    #             OPTIONS['cpc_model_path'] = encoder_path
    #             OPTIONS['which_module'] = which_module

    #             try:
    #                 main.run_configuration(
    #                     OPTIONS, f"no_pooling_linear_model_{encoder_name}_subset={subset_size}_epoch=790")
    #             except Exception as e:
    #                 # store to file
    #                 with open("error_log.txt", "a") as f:
    #                     f.write(f"*********Trying {encoder_name} subset={subset_size} module={which_module} *********\n")
    #                     f.write(str(e))
    #                     f.write("\n")
