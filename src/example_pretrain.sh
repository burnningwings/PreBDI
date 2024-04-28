#!/bin/sh
cd /data1/home/liujun/code/mvts_transformer/src
/data1/home/liujun/code/mvts_transformer/venv/bin/python \
main.py \
--output_dir PlanarBeamEX/pretrain \
--comment "pretraining PlanarBeam through imputation" \
--name PlanarBeam_pretrained_Crossformer \
--records_file PlanarBeamEX/pretrain/PlanarBeam_pretrained_Crossformer/Imputation_records.xls \
--data_dir data/PlanarBeam/ \
--data_class bridge \
--pattern train \
--val_pattern eval \
--test_pattern test \
--exclude_feats "1" \
--epochs 300 \
--lr 0.001 \
--optimizer RAdam \
--batch_size 64 \
--pos_encoding learnable \
--d_model 128 \
--num_nodes 32 \
--cross_data_dim 32 \
