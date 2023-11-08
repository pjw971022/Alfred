PROJECT_PATH=/home/pjw971022/RL/ConstGym/Grounding_LLMs_with_online_RL/experiments/configs/local_gpu_config.yaml
YOUR_OUTPUT_DIR=/home/pjw971022/RL/ConstGym/Grounding_LLMs_with_online_RL/outputs
PATH_TO_YOUR_LLM=t5-small 
HYDRA_FULL_ERROR=1 python -m lamorel_launcher.launch \
       --config-path $PROJECT_PATH/examples/PPO_LoRA_finetuning/ \
       --config-name local_gpu_config \
       rl_script_args.path=$PROJECT_PATH/examples/PPO_LoRA_finetuning/main.py \
       lamorel_args.accelerate_args.machine_rank=0 \
       rl_script_args.output_dir=$YOUR_OUTPUT_DIR \
       lamorel_args.llm_args.model_path=$PATH_TO_YOUR_LLM
