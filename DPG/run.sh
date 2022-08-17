CUDA_VISIBLE_DEVICES=2 nohup python run_simple.py --eval_num 5 --reinforce no --num_epochs 31 --train no &
CUDA_VISIBLE_DEVICES=2 nohup python run_simple.py --eval_num 5 --reinforce yes --num_epochs 31 --train no &
# CUDA_VISIBLE_DEVICES=0 nohup python run_simple.py --eval_num 5 --reinforce yes --num_epochs 200 --lr 0.001 &
# CUDA_VISIBLE_DEVICES=0 nohup python run_simple.py --eval_num 5 --reinforce yes --num_epochs 200 --lr 0.0005 &
# CUDA_VISIBLE_DEVICES=1 nohup python run_simple.py --eval_num 5 --reinforce yes --num_epochs 200 --lr 0.0001 &
# CUDA_VISIBLE_DEVICES=1 nohup python run_simple.py --eval_num 5 --reinforce yes --num_epochs 200 --lr 0.00005 &
# CUDA_VISIBLE_DEVICES=2 nohup python run.py --eval_num 5 --reinforce no --alpha 1.0 --beta 0.0 --lr 0.001 --train no &
# CUDA_VISIBLE_DEVICES=1 nohup python run.py --eval_num 5 --reinforce yes --alpha 0.2 --beta 0.0 --lr 0.001 &
# CUDA_VISIBLE_DEVICES=1 nohup python run.py --eval_num 5 --reinforce yes --alpha 0.2 --beta 0.1 --lr 0.001 &
# CUDA_VISIBLE_DEVICES=2 nohup python run.py --eval_num 5 --reinforce yes --alpha 0.2 --beta 0.2 &
# CUDA_VISIBLE_DEVICES=1 nohup python run.py --eval_num 5 --reinforce yes --alpha 0.0 --beta 0.0 --lr 0.0001 &
# CUDA_VISIBLE_DEVICES=1 nohup python run.py --eval_num 5 --reinforce yes --alpha 0.0 --beta 0.0 --lr 0.0005 &
# CUDA_VISIBLE_DEVICES=2 nohup python run.py --eval_num 5 --reinforce yes --alpha 0.0 --beta 0.0 --lr 0.00005 &
# CUDA_VISIBLE_DEVICES=2 nohup python run.py --eval_num 5 --reinforce yes --alpha 0.0 --beta 0.0 --lr 0.01 &
# CUDA_VISIBLE_DEVICES=1 nohup python run_loop.py --lr 0.001 --eval_num 5 --reinforce no --alpha 0.0 &

# CUDA_VISIBLE_DEVICES=2 python run.py --eval_num 5 --reinforce yes --alpha 0.0 --beta 0.0 --lr 0.01 --train no