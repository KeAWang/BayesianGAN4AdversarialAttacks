#python bayesian_gan_hmc.py --out_dir 'results' --semi_supervised --save_weights --save_samples --adv_test --train_iter 800 --data_path '' --n_save 10 --num_mcmc 4
python bayesian_gan_hmc.py --out_dir 'results' --semi_supervised --save_weights --save_samples --adv_test --train_iter 1000 --data_path '' --n_save 100  --num_mcmc 2  --numz 10 --adv_training
