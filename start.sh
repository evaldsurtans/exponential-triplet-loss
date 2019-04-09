#!/bin/sh -v

module load conda
export TMPDIR=$HOME/tmp
source activate conda_env
cd ~/Documents/fassion_minst/

#TODO - gala classification + male/female info

# TODO
# EMNIST - hand written
# CIFAR-100
# CIFAR-10
# https://pytorch.org/docs/stable/torchvision/models.html

# CARS https://ai.stanford.edu/~jkrause/cars/car_dataset.html
# https://arxiv.org/pdf/1511.06452.pdf
# Birds http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
# fassion http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html
# Online products dataset http://cvgl.stanford.edu/projects/lifted_struct/

#TODO very good paper https://arxiv.org/pdf/1703.07737.pdf
#TODO histogram loss https://arxiv.org/pdf/1611.00822.pdf
#TODO cosine softmax - alternative https://elib.dlr.de/116408/1/WACV2018.pdf

#TODO pielikt galā softmax

# publication https://arxiv.org/abs/1511.06452
# http://openaccess.thecvf.com/content_ECCV_2018/papers/Baosheng_Yu_Correcting_the_Triplet_ECCV_2018_paper.pdf
# hierarchical https://arxiv.org/pdf/1810.06951.pdf

# -params_grid tf_data_type split_avg_pool_size is_triplet_loss_margin_auto \
# https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Liu_DeepFashion_Powering_Robust_CVPR_2016_paper.pdf
# -params_grid triplet_loss \

# !!! WARNING when running on CPU hpc_gpu_process_count

# -datasource_exclude_train_class_ids 31 \
# -filter_samples none abs_margin \
# 114
# -params_grid learning_rate \
# cifar_10
#-datasource_type cifar_10 \
# -pre_type densenet121 resnet18 resnet34 \

# -embedding_layers 0 == fully convolutional

# eminst 31 == "V" (in between U, 4, 7)
# eminst 9 == "9" (maybe y, g)
# -datasource_exclude_train_class_ids 9 31 \

#-batch_size 76 44 128 152 \
#-triplet_positives 4 \

#-batch_size 95 55 160 \
#-triplet_positives 5 \

# 57 33 96 114


# v100 k40 8 12

python taskgen.py -repeat 1 -hpc_feautre_gpu v100 -hpc_queue batch -hpc_gpu_process_count 4 \
-hpc_gpu_count 1 -hpc_cpu_count_for_gpu 8 -hpc_cpu_count 32 -hpc_gpu_max_queue 9999 -device cuda \
-report apr_9_exp10_fassion_norm_single \
-batch_size 33 \
-triplet_positives 3 \
-optimizer adam \
-params_grid embedding_layers_last_norm learning_rate overlap_coef \
-learning_rate 1e-3 1e-4 3e-3 \
-overlap_coef 1.0 3.0 1.5 \
-embedding_layers_last_norm none instance batch layer local \
-max_embeddings_per_class_test 0 \
-max_embeddings_per_class_train 0 \
-max_embeddings_histograms 0 \
-slope_coef 1.0 \
-neg_coef 2.0 \
-pos_coef 2.0 \
-triplet_loss exp10 \
-embedding_layers 1 \
-embedding_layers_hidden_func relu \
-embedding_layers_hidden 512 \
-leaky_relu_slope 0.01 \
-datasource_type fassion_minst \
-embedding_size 32 \
-embedding_function tanh \
-conv_expansion_rate 2 \
-conv_first_channel_count 32 \
-conv_first_kernel 7 \
-conv_kernel 5 \
-conv_resnet_layers 4 \
-conv_resnet_sub_layers 3 \
-is_conv_max_pool False \
-triplet_sampler_var all \
-is_center_loss False \
-is_kl_loss False \
-kl_coef 1e-4 \
-coef_loss_neg 1.0 \
-lossless_beta 1.2 \
-embedding_norm unit_range \
-triplet_similarity cos \
-filter_samples none \
-is_triplet_loss_margin_auto False \
-triplet_loss_margin 0.2 \
-triplet_sampler triplet_sampler_5_zipper  \
-model model_10_cold \
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



