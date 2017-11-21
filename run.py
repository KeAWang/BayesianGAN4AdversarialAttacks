import os

epoch_size = 1
num_epochs = 2
experiment_name = "adv_train_only"

x = """python bayesian_gan_hmc.py --out_dir \'results\' --semi_supervised --save_weights --save_samples --adv_test --data_path \'\' --save_chkpt 1 --n_save 100 --num_mcmc 1  --numz 1 --custom_experiment """+experiment_name+" "

base_dir ="--chkpt ~/ORIE6741_bayesgan/results/"+experiment_name+"/"

for epoch in range(1,num_epochs+1):
    starting_iter = str((epoch-1)*epoch_size)
    ending_iter = str(epoch*epoch_size)
    if epoch==1:
        os.system(x+" --train_iter "+ending_iter)
    else:
        os.system(x+base_dir+"model_"+starting_iter + " --load_chkpt" + " --train_iter "+ending_iter)
