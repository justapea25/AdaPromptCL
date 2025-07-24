CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
  --nproc_per_node=1 --master_port=12345 --use_env main.py \
  cifar100_dualprompt \
  --model vit_base_patch16_224 \
  --batch-size 64 \
  --data-path "/home/hoangminhdau/work/Working/AdaPromptCL/data" \
  --output_dir "./output" \
  --use_e_prompt --e_prompt_layer_idx 0 1 2 3 4 \
  --seed 0

  python main.py deepfake_dualprompt \
  --model vit_base_patch16_224 \
  --batch-size 32 \
  --epochs 5 \
  --data-path "/path/to/your/data" \
  --output_dir "./output" \
  --deepfake_tasks gaugan biggan stylegan stargan \
  --num_tasks 4 \
  --use_e_prompt --e_prompt_layer_idx 0 1 2 3 4 \
  --seed 42 \
  --data_driven_evolve --uni_or_specific --converge_thrh 0.4 \
  --mergable_prompts --postmerge_thrh 0.6