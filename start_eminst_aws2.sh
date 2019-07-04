#!/bin/sh -v

module load conda
export TMPDIR=$HOME/tmp
source activate conda_env
cd ~/Documents/fassion_minst/


python taskgen_linux.py -repeat 1 \
-device cuda \
-report jul_4_model_12_dobe_aws_2_tensor_ver \
-batch_size 33 \
-triplet_positives 3 \
-epochs_count 100 \
-datasource_type eminst \
-optimizer adam \
-params_grid learning_rate overlap_coef center_loss_coef pos_loss_coef center_loss_min_count \
-center_loss_min_count 300 100 \
-learning_rate 1e-4 3e-5 \
-is_center_loss True \
-is_class_loss True \
-pos_loss_coef 2.0 1.0 \
-neg_loss_coef 1.0 \
-center_loss_coef 2.0 1.0 \
-class_loss_coef 1.0 \
-embedding_init zeros \
-overlap_coef 1.0 20.0 40.0 60.0 \
-embedding_norm unit_range \
-embedding_scale 2.0 \
-triplet_similarity euclidean \
-layers_embedding_dropout 0.0 \
-layers_embedding_type last \
-embedding_layers 0 \
-suffix_affine_layers_hidden_func maxout \
-embedding_layers_hidden 512 \
-suffix_affine_layers_hidden_params 16 \
-is_model_encoder_pretrained True \
-model_encoder densenet161 \
-embedding_size 128 \
-embedding_layers_last_norm none \
-max_embeddings_per_class_test 0 \
-max_embeddings_per_class_train 0 \
-max_embeddings_histograms 0 \
-slope_coef 1.0 \
-pos_coef 0.0 \
-neg_coef 0.0 \
-triplet_loss exp13 \
-embedding_layers_hidden_func relu \
-leaky_relu_slope 0.01 \
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
-filter_samples none \
-is_triplet_loss_margin_auto False \
-triplet_loss_margin 0.2 \
-triplet_sampler triplet_sampler_4 \
-model model_12_dobe \
-is_pre_grad_locked False \
-datasource datasource_pytorch \
-is_hpc False \
-is_single_cuda_device True \
-local_process_count_per_task 8 \
-single_task False \
-is_quick_test False


# euclidean unit_range
# cos l2
# exp8

# -triplet_loss exp1 standard standard2 lossless lifted lifted2 \
# speaker_small_male_4000_log_dual_13
# speaker_small_female_4000_log_dual_13


