python train_simple.py --config config_simple.yaml

python inference.py \
--checkpoint checkpoints/best_model.pt \
--mode all_cities \
--image-base-dir /root/lanyun-fs/project/my_seq/data/satellite_images \
--text-base-dir /root/lanyun-fs/project/my_seq/data/llm_texts \
--output-dir embeddings_output
