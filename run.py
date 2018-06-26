import os


"""
epoch_size = 200
num_epochs = 20
start_epoch = 6
experiment_name = "adv_test_only"

x = "python bayesian_gan_hmc.py --out_dir \'results\' --semi_supervised --save_weights --save_samples --adv_test --data_path \'\' --n_save 100 --num_mcmc 2  --numz 10 --custom_experiment "+experiment_name+" --save_chkpt "+str(epoch_size)

base_dir =" --chkpt ~/ORIE6741_bayesgan/results/"+experiment_name+"/"

for epoch in range(start_epoch,num_epochs+1):
    starting_iter = str((epoch-1)*epoch_size)
    ending_iter = str(epoch*epoch_size)
    penultimate_iter = str((epoch-2)*epoch_size)
    if epoch==1:
        os.system(x+" --train_iter "+ending_iter)
    else:
        os.system("rm ~/ORIE6741_bayesgan/results/"+experiment_name+"/model_"+penultimate_iter+".*")
        os.system(x+base_dir+"model_"+starting_iter + " --load_chkpt" + " --train_iter "+ending_iter)
"""

"""
experiment_name = "early_training"
os.system("python bayesian_gan_hmc.py --out_dir \'results\' --semi_supervised --save_samples --adv_test --data_path \'\' --n_save 10 --num_mcmc 2 --numz 10 --custom_experiment "+experiment_name+" --train_iter 150")
"""

"""
epoch_size = 200
num_epochs = 20
start_epoch = 1 
experiment_name = "mcmc-1"

x = "python bayesian_gan_hmc.py --out_dir \'results\' --semi_supervised --save_samples --adv_test --data_path \'\' --n_save 100 --num_mcmc 1  --numz 10 --custom_experiment "+experiment_name+" --save_chkpt "+str(epoch_size)

base_dir =" --chkpt ~/ORIE6741_bayesgan/results/"+experiment_name+"/"

for epoch in range(start_epoch,num_epochs+1):
    starting_iter = str((epoch-1)*epoch_size)
    ending_iter = str(epoch*epoch_size)
    penultimate_iter = str((epoch-2)*epoch_size)
    if epoch==1:
        os.system(x+" --train_iter "+ending_iter)
    else:
        os.system("rm ~/ORIE6741_bayesgan/results/"+experiment_name+"/model_"+penultimate_iter+".*")
        os.system(x+base_dir+"model_"+starting_iter + " --load_chkpt" + " --train_iter "+ending_iter)
"""

"""
experiment_name = "MLE_DCGAN"
num_iters = 4000

x = "python bayesian_gan_hmc.py --out_dir \'results\' --semi_supervised --save_samples --ml_ensemble 1 --dataset mnist --adv_test --data_path \'\' --n_save 100 --num_mcmc 2  --numz 10 --custom_experiment "+experiment_name

os.system(x+" --train_iter "+str(num_iters))
"""


"""
epoch_size = 400
num_epochs = 5 
start_epoch = 1
experiment_name = "mcmc-4"

x = "python bayesian_gan_hmc.py --out_dir \'results\' --semi_supervised --save_samples --adv_test --data_path \'\' --n_save 100 --num_mcmc 4  --numz 10 --custom_experiment "+experiment_name+" --save_chkpt "+str(epoch_size)

base_dir =" --chkpt ~/ORIE6741_bayesgan/results/"+experiment_name+"/"

for epoch in range(start_epoch,num_epochs+1):
    starting_iter = str((epoch-1)*epoch_size)
    ending_iter = str(epoch*epoch_size)
    penultimate_iter = str((epoch-2)*epoch_size)
    if epoch==1:
        os.system(x+" --train_iter "+ending_iter)
    else:
        os.system(x+base_dir+"model_"+starting_iter + " --load_chkpt" + " --train_iter "+ending_iter
                 +" && "
                 +"rm ~/ORIE6741_bayesgan/results/"+experiment_name+"/model_"+penultimate_iter+".*")
"""
"""
experiment_name = "MLE_DCGAN_basic_iterative"
num_iters = 2000
x = "python bayesian_gan_hmc.py --out_dir \'results\' --semi_supervised --save_samples --ml_ensemble 1 --dataset mnist --adv_test --basic_iterative --data_path \'\' --n_save 100  --numz 10 --custom_experiment "+experiment_name 
os.system(x+" --train_iter "+str(num_iters))
"""

"""
epoch_size = 400
num_epochs = 5 
start_epoch = 1
experiment_name = "mcmc-4_basic_iterative"

x = "python bayesian_gan_hmc.py --out_dir \'results\' --semi_supervised --save_samples --adv_test --basic_iterative --data_path \'\' --n_save 100 --num_mcmc 4  --numz 10 --custom_experiment "+experiment_name+" --save_chkpt "+str(epoch_size)

base_dir =" --chkpt ~/ORIE6741_bayesgan/results/"+experiment_name+"/"

for epoch in range(start_epoch,num_epochs+1):
    starting_iter = str((epoch-1)*epoch_size)
    ending_iter = str(epoch*epoch_size)
    penultimate_iter = str((epoch-2)*epoch_size)
    if epoch==1:
        os.system(x+" --train_iter "+ending_iter)
    else:
        os.system(x+base_dir+"model_"+starting_iter + " --load_chkpt" + " --train_iter "+ending_iter
                 +" && "
                 +"rm ~/ORIE6741_bayesgan/results/"+experiment_name+"/model_"+penultimate_iter+".*")
"""

"""
epoch_size = 400
num_epochs = 5 
start_epoch = 1
experiment_name = "mcmc-2_basic_iterative"

x = "python bayesian_gan_hmc.py --out_dir \'results\' --semi_supervised --save_samples --adv_test --basic_iterative --data_path \'\' --n_save 100 --num_mcmc 2  --numz 10 --custom_experiment "+experiment_name+" --save_chkpt "+str(epoch_size)

base_dir =" --chkpt ~/ORIE6741_bayesgan/results/"+experiment_name+"/"

for epoch in range(start_epoch,num_epochs+1):
    starting_iter = str((epoch-1)*epoch_size)
    ending_iter = str(epoch*epoch_size)
    penultimate_iter = str((epoch-2)*epoch_size)
    if epoch==1:
        os.system(x+" --train_iter "+ending_iter)
    else:
        os.system(x+base_dir+"model_"+starting_iter + " --load_chkpt" + " --train_iter "+ending_iter
                 +" && "
                 +"rm ~/ORIE6741_bayesgan/results/"+experiment_name+"/model_"+penultimate_iter+".*")
"""


epoch_size = 400
num_epochs = 5 
start_epoch = 1
experiment_name = "example_adv_imgs"

x = "python bayesian_gan_hmc.py --out_dir \'results\' --semi_supervised --save_samples --adv_test --basic_iterative --data_path \'\' --n_save 1 --num_mcmc 1  --numz 1 --custom_experiment "+experiment_name)

base_dir =" --chkpt ~/ORIE6741_bayesgan/results/"+experiment_name+"/"

for epoch in range(start_epoch,num_epochs+1):
    starting_iter = str((epoch-1)*epoch_size)
    ending_iter = str(epoch*epoch_size)
    penultimate_iter = str((epoch-2)*epoch_size)
    os.system(x+" --train_iter "+ending_iter)
