#!/bin/sh -v

module load conda
export TMPDIR=$HOME/tmp
source activate conda_env
cd ~/Documents/fassion_minst/


# 1024!!!!!!!
# center_loss_min_count
#noise_training
python taskgen.py -repeat 1 -hpc_feautre_gpu v100 -hpc_queue batch -hpc_gpu_process_count 4 \
-hpc_gpu_count 1 -hpc_cpu_count_for_gpu 8 -hpc_cpu_count 12 -hpc_gpu_max_queue 9999 -device cuda \
-report jun_7_model_12_dobe_exp13_cifar_100_noise \
-batch_size 33 \
-triplet_positives 3 \
-epochs_count 300 \
-datasource_type cifar_100 \
-optimizer adam \
-params_grid overlap_coef pos_loss_coef center_loss_coef class_loss_coef pos_loss_coef \
-learning_rate 1e-4 \
-early_stopping_delta_percent 0.005 \
-learning_rate_min 0.0 \
-is_center_loss True \
-noise_training 0.5 \
-is_class_loss True \
-pos_loss_coef 1.0 0.0 \
-neg_loss_coef 1.0 \
-center_loss_coef 1.0 2.0 \
-class_loss_coef 0.0 1.0 2.0 \
-center_loss_min_count 100 \
-overlap_coef 10.0 20.0 30.0 40.0 \
-layers_embedding_dropout 0.0 \
-layers_embedding_type last \
-embedding_layers 0 \
-embedding_layers_hidden_func relu \
-embedding_layers_hidden 512 \
-suffix_affine_layers_hidden_func maxout \
-suffix_affine_layers_hidden_params 16 \
-is_model_encoder_pretrained True \
-model_encoder densenet161 \
-embedding_layers_last_norm none \
-max_embeddings_per_class_test 0 \
-max_embeddings_per_class_train 0 \
-max_embeddings_histograms 0 \
-slope_coef 1.0 \
-pos_coef 0.0 \
-neg_coef 0.0 \
-triplet_loss exp13 \
-leaky_relu_slope 0.01 \
-embedding_size 1024 \
-embedding_function tanh \
-conv_expansion_rate 2 \
-conv_first_channel_count 32 \
-conv_first_kernel 7 \
-conv_kernel 5 \
-conv_resnet_layers 4 \
-conv_resnet_sub_layers 3 \
-is_conv_max_pool False \
-triplet_sampler_var hard \
-is_kl_loss False \
-kl_coef 1e-4 \
-coef_loss_neg 1.0 \
-lossless_beta 1.2 \
-embedding_norm unit_range \
-triplet_similarity euclidean \
-filter_samples none \
-is_triplet_loss_margin_auto False \
-triplet_loss_margin 0.2 \
-triplet_sampler triplet_sampler_5_zipper  \
-model model_12_dobe \
-is_pre_grad_locked False \
-datasource datasource_pytorch \
-is_hpc True \
-is_quick_test False \
-single_task False


# euclidean unit_range
# cos l2
# exp8

# -triplet_loss exp1 standard standard2 lossless lifted lifted2 \
# speaker_small_male_4000_log_dual_13
# speaker_small_female_4000_log_dual_13



