python -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('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')"
HYDRA_FULL_ERROR=1
LOGLEVEL=WARNING python -m lamorel_launcher.launch \
                        --config-path /home/pjw971022/RL/ConstGym/Grounding_LLMs_with_online_RL/lamorel/examples/SayCan \
                        --config-name local_gpu_config \
                        rl_script_args.path=/home/pjw971022/RL/ConstGym/Grounding_LLMs_with_online_RL/lamorel/examples/SayCan/run_pickplace_saycan.py \
                        lamorel_args.accelerate_args.machine_rank=0 