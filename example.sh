CUDA_id=$1


CUDA_VISIBLE_DEVICES=${CUDA_id} python main.py \
--dataset blogcatalog \
--tau 0.05 \
--co_lambda 0.1 \
--alpha 1 \
--lr 0.001 \
--epochs 200 \
--K 100 \
--n_neg 100 \
--th 0.95 \
--ptb_rate 0.3 \
--noise uniform \
--decay_w 0.1 

