#!/bin/sh
cd /home/liujun/BHM/mvts_transformer/src
/home/liujun/BHM/mvts_transformer/venv/bin/python \
main.py \
--output_dir BenchmarkEX/Classification \
--comment "Benchmark domain finetune for classification " \
--name BenchmarkEXP_PreBDI \
--records_file BenchmarkEX/Classification/Benchmark_PreBDI/classification_records.xls \
--data_dir data/BenchmarkEXP/ \
--data_class bridge \
--pattern train \
--val_pattern eval \
--test_pattern eval \
--normalization None \
--epochs 200 \
--lr 0.0001 \
--optimizer Adam \
--pos_encoding learnable \
--d_model 64 \
--num_nodes 16 \
--cross_data_dim 16 \
--input_len 128 \
--embed_dim 128 \
--node_dim 128 \
--output_len 9 \
--cross_in_len 128 \
--task classification \
--change_output \
--batch_size 128 \
--val_interval 1 \
--model CFEACL