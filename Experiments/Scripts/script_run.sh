# Script to run main.py with slurm
# Usage: ./script_run.sh

# nohup srun --job-name=n1K --nodelist 'deepthought' --mem=20000 --gres=gpu:1 --cpus-per-task=2 python -u main.py > ../Logs/n1K disown &
# nohup srun --job-name=n2K --nodelist 'deepthought' --mem=20000 --gres=gpu:1 --cpus-per-task=2 python -u main.py > ../Logs/n2K disown &
# nohup srun --job-name=n4K --nodelist 'deepthought' --mem=20000 --gres=gpu:1 --cpus-per-task=2 python -u main.py > ../Logs/n4K disown &
# nohup srun --job-name=n8K --nodelist 'deepthought' --mem=20000 --gres=gpu:1 --cpus-per-task=2 python -u main.py > ../Logs/n8K disown &
# nohup srun --job-name=n16K --nodelist 'deepthought' --mem=20000 --gres=gpu:1 --cpus-per-task=2 python -u main.py > ../Logs/n16K disown &
# nohup srun --job-name=n32K --nodelist 'deepthought' --mem=20000 --gres=gpu:1 --cpus-per-task=2 python -u main.py > ../Logs/n32K disown &


# ------------------ SGD ------------------
# # d = 8
# nohup srun --job-name=n256 --nodelist 'diffusion' --mem=20000 --gres=gpu:1 --cpus-per-task=4 python -u run_GMM.py -n 256 -d 8 -s 1 -de 128 -O 'SGD_Momentum' > ../Logs/n256 &
# nohup srun --job-name=n512 --nodelist 'diffusion' --mem=20000 --gres=gpu:1 --cpus-per-task=4 python -u run_GMM.py -n 512 -d 8 -s 1 -de 128 -O 'SGD_Momentum' > ../Logs/n512 &
# nohup srun --job-name=n1K --nodelist 'diffusion' --mem=20000 --gres=gpu:1 --cpus-per-task=4 python -u run_GMM.py -n 1024 -d 8 -s 1 -de 128 -O 'SGD_Momentum' > ../Logs/n1K &
# nohup srun --job-name=n2K --nodelist 'diffusion' --mem=20000 --gres=gpu:1 --cpus-per-task=4 python -u run_GMM.py -n 2048 -d 8 -s 1 -de 128 -O 'SGD_Momentum' > ../Logs/n2K &
# nohup srun --job-name=n4K --nodelist 'diffusion' --mem=20000 --gres=gpu:1 --cpus-per-task=4 python -u run_GMM.py -n 4096 -d 8 -s 1 -de 128 -O 'SGD_Momentum' > ../Logs/n4K &
# # nohup srun --job-name=n8K --nodelist 'diffusion' --mem=20000 --gres=gpu:1 --cpus-per-task=4 python -u run_GMM.py -n 8192 -d 8 -s 1 -de 128 > ../Logs/n8K &
# # nohup srun --job-name=n16K --nodelist 'diffusion' --mem=20000 --gres=gpu:1 --cpus-per-task=4 python -u run_GMM.py -n 16384 -d 8 -s 1 -de 128 > ../Logs/n16K &
# # nohup srun --job-name=n32K --nodelist 'diffusion' --mem=20000 --gres=gpu:1 --cpus-per-task=4 python -u run_GMM.py -n 32768 -d 8 -s 1 -de 128 > ../Logs/n32K &


# # d = 16
# nohup srun --job-name=n256_d16 --nodelist 'diffusion' --mem=20000 --gres=gpu:1 --cpus-per-task=4 python -u run_GMM.py -n 256 -d 16 -s 1 -de 128 -O 'SGD_Momentum' > ../Logs/n256_d16 &
# nohup srun --job-name=n512_d16 --nodelist 'diffusion' --mem=20000 --gres=gpu:1 --cpus-per-task=4 python -u run_GMM.py -n 512 -d 16 -s 1 -de 128 -O 'SGD_Momentum' > ../Logs/n512_d16 &
# nohup srun --job-name=n1K_d16 --nodelist 'diffusion' --mem=20000 --gres=gpu:1 --cpus-per-task=4 python -u run_GMM.py -n 1024 -d 16 -s 1 -de 128 -O 'SGD_Momentum' > ../Logs/n1K_d16 &
# nohup srun --job-name=n2K_d16 --nodelist 'diffusion' --mem=20000 --gres=gpu:1 --cpus-per-task=4 python -u run_GMM.py -n 2048 -d 16 -s 1 -de 128 -O 'SGD_Momentum' > ../Logs/n2K_d16 &
# nohup srun --job-name=n4K_d16 --nodelist 'diffusion' --mem=20000 --gres=gpu:1 --cpus-per-task=4 python -u run_GMM.py -n 4096 -d 16 -s 1 -de 128 -O 'SGD_Momentum' > ../Logs/n4K_d16 &


# # d = 32
# nohup srun --job-name=n256_d32 --nodelist 'diffusion' --mem=20000 --gres=gpu:1 --cpus-per-task=4 python -u run_GMM.py -n 256 -d 32 -s 1 -de 128 -O 'SGD_Momentum' > ../Logs/n256_d32 &
# nohup srun --job-name=n512_d32 --nodelist 'diffusion' --mem=20000 --gres=gpu:1 --cpus-per-task=4 python -u run_GMM.py -n 512 -d 32 -s 1 -de 128 -O 'SGD_Momentum' > ../Logs/n512_d32 &
# nohup srun --job-name=n1K_d32 --nodelist 'diffusion' --mem=20000 --gres=gpu:1 --cpus-per-task=4 python -u run_GMM.py -n 1024 -d 32 -s 1 -de 128 -O 'SGD_Momentum' > ../Logs/n1K_d32 &
# nohup srun --job-name=n2K_d32 --nodelist 'diffusion' --mem=20000 --gres=gpu:1 --cpus-per-task=4 python -u run_GMM.py -n 2048 -d 32 -s 1 -de 128 -O 'SGD_Momentum' > ../Logs/n2K_d32 &
# nohup srun --job-name=n4K_d32 --nodelist 'diffusion' --mem=20000 --gres=gpu:1 --cpus-per-task=4 python -u run_GMM.py -n 4096 -d 32 -s 1 -de 128 -O 'SGD_Momentum' > ../Logs/n4K_d32 &




# # ------------------ Adam ------------------
# # d = 8
# nohup srun --job-name=n256_A --nodelist 'diffusion' --mem=20000 --gres=gpu:1 --cpus-per-task=4 python -u run_GMM.py -n 256 -d 8 -s 1 -de 128 -O 'Adam' > ../Logs/n256_A &
# nohup srun --job-name=n512_A --nodelist 'diffusion' --mem=20000 --gres=gpu:1 --cpus-per-task=4 python -u run_GMM.py -n 512 -d 8 -s 1 -de 128 -O 'Adam' > ../Logs/n512_A &
# nohup srun --job-name=n1K_A --nodelist 'diffusion' --mem=20000 --gres=gpu:1 --cpus-per-task=4 python -u run_GMM.py -n 1024 -d 8 -s 1 -de 128 -O 'Adam' > ../Logs/n1K_A &
# nohup srun --job-name=n2K_A --nodelist 'diffusion' --mem=20000 --gres=gpu:1 --cpus-per-task=4 python -u run_GMM.py -n 2048 -d 8 -s 1 -de 128 -O 'Adam' > ../Logs/n2K_A &
# nohup srun --job-name=n4K_A --nodelist 'diffusion' --mem=20000 --gres=gpu:1 --cpus-per-task=4 python -u run_GMM.py -n 4096 -d 8 -s 1 -de 128 -O 'Adam' > ../Logs/n4K_A &

# # d = 16
# nohup srun --job-name=n256_d16_A --nodelist 'diffusion' --mem=20000 --gres=gpu:1 --cpus-per-task=4 python -u run_GMM.py -n 256 -d 16 -s 1 -de 128 -O 'Adam' > ../Logs/n256_d16_A &
# nohup srun --job-name=n512_d16_A --nodelist 'diffusion' --mem=20000 --gres=gpu:1 --cpus-per-task=4 python -u run_GMM.py -n 512 -d 16 -s 1 -de 128 -O 'Adam' > ../Logs/n512_d16_A &
# nohup srun --job-name=n1K_d16_A --nodelist 'diffusion' --mem=20000 --gres=gpu:1 --cpus-per-task=4 python -u run_GMM.py -n 1024 -d 16 -s 1 -de 128 -O 'Adam' > ../Logs/n1K_d16_A &
# nohup srun --job-name=n2K_d16_A --nodelist 'diffusion' --mem=20000 --gres=gpu:1 --cpus-per-task=4 python -u run_GMM.py -n 2048 -d 16 -s 1 -de 128 -O 'Adam' > ../Logs/n2K_d16_A &
# nohup srun --job-name=n4K_d16_A --nodelist 'diffusion' --mem=20000 --gres=gpu:1 --cpus-per-task=4 python -u run_GMM.py -n 4096 -d 16 -s 1 -de 128 -O 'Adam' > ../Logs/n4K_d16_A &


# # d = 32
# nohup srun --job-name=n256_d32_A --nodelist 'diffusion' --mem=20000 --gres=gpu:1 --cpus-per-task=4 python -u run_GMM.py -n 256 -d 32 -s 1 -de 128 -O 'Adam' > ../Logs/n256_d32_A &
# nohup srun --job-name=n512_d32_A --nodelist 'diffusion' --mem=20000 --gres=gpu:1 --cpus-per-task=4 python -u run_GMM.py -n 512 -d 32 -s 1 -de 128 -O 'Adam' > ../Logs/n512_d32_A &
# nohup srun --job-name=n1K_d32_A --nodelist 'diffusion' --mem=20000 --gres=gpu:1 --cpus-per-task=4 python -u run_GMM.py -n 1024 -d 32 -s 1 -de 128 -O 'Adam' > ../Logs/n1K_d32_A &
# nohup srun --job-name=n2K_d32_A --nodelist 'diffusion' --mem=20000 --gres=gpu:1 --cpus-per-task=4 python -u run_GMM.py -n 2048 -d 32 -s 1 -de 128 -O 'Adam' > ../Logs/n2K_d32_A &
# nohup srun --job-name=n4K_d32_A --nodelist 'diffusion' --mem=20000 --gres=gpu:1 --cpus-per-task=4 python -u run_GMM.py -n 4096 -d 32 -s 1 -de 128 -O 'Adam' > ../Logs/n4K_d32_A &

