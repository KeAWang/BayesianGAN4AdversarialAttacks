import os
import sys
import argparse
import json

import numpy as np

import tensorflow as tf
from tensorflow.contrib import slim

from bgan_util import AttributeDict
from bgan_util import print_images, MnistDataset, CelebDataset, Cifar10, SVHN, ImageNet
from bgan_models import BDCGAN

from bayesian_gan_hmc import *

sys.path.insert(0, '/home/ubuntu/cleverhans')
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_tf import model_train, model_eval,model_loss
from cleverhans.model import Model

import time


def get_test_stats(session, dcgan, all_test_img_batches, all_test_lbls):

    # only need this function because bdcgan has a fixed batch size for *everything*
    # test_size is in number of batches
    all_d_logits, all_s_logits = [], []
    for test_image_batch, test_lbls in zip(all_test_img_batches, all_test_lbls):
        test_d_logits, test_s_logits = session.run([dcgan.test_D_logits, dcgan.test_S_logits],
                                                   feed_dict={dcgan.test_inputs: test_image_batch})
        all_d_logits.append(test_d_logits)
        all_s_logits.append(test_s_logits)

    test_d_logits = np.concatenate(all_d_logits)
    test_s_logits = np.concatenate(all_s_logits)
    test_lbls = np.concatenate(all_test_lbls)

    return test_d_logits, test_s_logits, test_lbls


def ml_dcgan(dataset, args):

    z_dim = args.z_dim
    x_dim = dataset.x_dim
    batch_size = args.batch_size

    base_learning_rate = args.lr # for now we use same learning rate for Ds and Gs
    lr_decay_rate = args.lr_decay

    dataset_size = dataset.dataset_size

    print("Starting session")
    session = get_session()

    dcgan = BDCGAN(x_dim, z_dim,
                   dataset_size=dataset_size,
                   batch_size=batch_size,
                   J=1, ml=True,
                   num_classes=dataset.num_classes)

    tf.global_variables_initializer().run()

    print("Starting training loop")
        
    test_image_batches, test_label_batches = get_test_batches(dataset, batch_size)
    supervised_batches = get_supervised_batches(dataset, args.N, batch_size, list(range(dataset.num_classes)))

    if args.adv_test:
	
        x = tf.placeholder(tf.float32, shape=(batch_size, 28, 28, 1)) # Hardcoded for MNIST
        fgsm = FastGradientMethod(dcgan, sess=session)
        dcgan.adv_constructor = fgsm
        eval_params = {'batch_size': batch_size}
        fgsm_params = {'eps': args.eps,
                   'clip_min': 0.,
                   'clip_max': 1.}
        adv_x = fgsm.generate(x,**fgsm_params)
        preds = dcgan.get_probs(adv_x)

    for train_iter in range(args.train_iter):
        
        batch_z = np.random.uniform(-1, 1, [batch_size, z_dim])
        image_batch, _ = dataset.next_batch(batch_size, class_id=None)
        labeled_image_batches, label_batches = next(supervised_batches)

        learning_rate = base_learning_rate * np.exp(-lr_decay_rate *
                                                    min(1.0, (train_iter*batch_size)/float(dataset_size)))


        if args.adv_train:
            adv_labeled = session.run(fgsm.generate(labeled_image_ph,**fgsm_targeted_params), feed_dict = {labeled_image_ph:labeled_image_batch})
            adv_unlabeled = session.run(fgsm.generate(unlabeled_batch_ph,**fgsm_params),feed_dict = {unlabeled_batch_ph:image_batch})
            _, d_loss = session.run([dcgan.d_optim_semi, dcgan.d_loss_semi], feed_dict={dcgan.labeled_inputs: labeled_image_batch,
                                                                                        dcgan.labels: get_gan_labels(labels),
                                                                                        dcgan.inputs: image_batch,
                                                                                        dcgan.z: batch_z,
                                                                                        dcgan.d_semi_learning_rate: learning_rate,
                                                                                        dcgan.adv_unlab: adv_unlabeled,
                                                                                        dcgan.adv_labeled: adv_labeled
                                                                                        })
        else:
            _, d_loss = session.run([dcgan.d_optim_semi, dcgan.d_loss_semi], feed_dict={dcgan.labeled_inputs: labeled_image_batches,
                                                                                        dcgan.labels: get_gan_labels(label_batches),
                                                                                        dcgan.inputs: image_batch,
                                                                                        dcgan.z: batch_z,
                                                                                        dcgan.d_semi_learning_rate: learning_rate,

                                                                                       })

        _, s_loss = session.run([dcgan.s_optim, dcgan.s_loss], feed_dict={dcgan.inputs: labeled_image_batches,
                                                                          dcgan.lbls: label_batches})
        # compute g_sample loss
        batch_z = np.random.uniform(-1, 1, [batch_size, z_dim])
        _, g_loss = session.run([dcgan.g_optims[0], dcgan.generation["g_losses"][0]],
                                feed_dict={dcgan.z: batch_z,
                                           dcgan.g_learning_rate: learning_rate})

        if train_iter % args.n_save == 0:
            # get test set performance on real labels only for both GAN-based classifier and standard one
            d_logits, s_logits, lbls = get_test_stats(session, dcgan, test_image_batches, test_label_batches)

            if args.adv_test:
                adv_set = []
                for test_images in test_image_batches:
                    adv_set.append(session.run(adv_x, feed_dict = {x:test_images}))
                adv_sup_acc, adv_ss_acc,correct_uncertainty, incorrect_uncertainty, adv_acc, adv_ex_prob = get_adv_test_accuracy(session,dcgan,adv_set,test_label_batches)

            print("saving results")
            np.savez_compressed(os.path.join(args.out_dir, 'results_%i.npz' % train_iter),
                                d_logits=d_logits, s_logits=s_logits, lbls=lbls,
                                adv_sup_acc=adv_sup_acc, adv_sup_acc_unf=adv_ss_acc,
                                correct_uncertainty=correct_uncertainty,
                                incorrect_uncertainty=incorrect_uncertainty,
                                adv_acc=adv_acc)

            var_dict = {}
            for var in tf.trainable_variables():
                var_dict[var.name] = session.run(var.name)

            np.savez_compressed(os.path.join(args.out_dir,
                                             "weights_%i.npz" % train_iter),
                                **var_dict)
            

            print("done")

    print("closing session")
    session.close()
    tf.reset_default_graph()
