import sys


arguments = sys.argv[1:]
# eg: Python main.py abc def, then arguments = ['abc', 'def']

assert len(arguments) in [1, 2], \
    "At least one argument is required. The first argument is the experiment name and the second argument is the config file name."
# 1 arg for default, only provides the experiment_name
# 2 arg for custom config, provides the experiment_name and the config file name


experiment_name = arguments[0]

if len(arguments) == 1:
    # Default options
    from configs.enc_default import _get_options

else:
    file = arguments[1]
    # eg: `from configs.enc_gim_audio import get_options`
    exec(f'from configs.{file} import _get_options')

get_options = lambda: _get_options(experiment_name=experiment_name)
