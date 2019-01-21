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

# -datasource_exclude_train_class_ids 2 \
# -filter_samples none abs_margin \
# 114
# -params_grid learning_rate \
# cifar_10
#-datasource_type cifar_10 \

python taskgen.py -repeat 1 -hpc_feautre_gpu v100 -hpc_queue batch -hpc_gpu_process_count 4 \
-hpc_gpu_count 1 -hpc_cpu_count_for_gpu 8 -hpc_cpu_count 32 -hpc_gpu_max_queue 0 -device cpu \
-report test_19_rock_cifar \
-batch_size 114 \
-triplet_positives 3 \
-optimizer adam \
-params_grid learning_rate embedding_norm filter_samples \
-learning_rate 1e-3 1e-4 \
-datasource_type cifar_10 \
-embedding_size 32 \
-embedding_function tanh \
-suffix_affine_layers 2 \
-suffix_affine_layers_hidden 512 \
-conv_expansion_rate 2 \
-conv_first_channel_count 8 \
-conv_first_kernel 7 \
-conv_kernel 3 \
-conv_resnet_layers 2 \
-conv_resnet_sub_layers 3 \
-is_conv_max_pool False \
-exp_coef 2.0 \
-triplet_sampler_var hard \
-triplet_loss exp2 \
-coef_loss_neg 1.0 \
-lossless_beta 2.0 \
-triplet_loss_margin 0.2 \
-embedding_norm l2 none \
-triplet_similarity cos \
-filter_samples none abs_margin_asym \
-is_triplet_loss_margin_auto False \
-triplet_sampler triplet_sampler_4 \
-model model_pink_rock \
-datasource datasource_pytorch \
-early_stopping_patience 5 \
-epochs_count 30 -is_hpc True \
-is_quick_test False \
-single_task False

# euclidean

# -triplet_loss exp1 standard standard2 lossless lifted lifted2 \
# speaker_small_male_4000_log_dual_13
# speaker_small_female_4000_log_dual_13



