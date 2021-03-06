#!/usr/bin/env python

import os
import sys
import argparse
import json
import time
from datetime import datetime

import numpy as np
from math import ceil
from scipy.misc import imsave

from PIL import Image

import tensorflow as tf
from tensorflow.contrib import slim

from bgan_util import AttributeDict
from bgan_util import print_images, MnistDataset, CelebDataset, Cifar10, SVHN, ImageNet
from bgan_models import BDCGAN
import sys

#sys.path.insert(0, '/home/ubuntu/cleverhans')
#sys.path.insert(0, '/Users/mattwallingford/Documents/cleverhans')
sys.path.insert(0, '/home/alex/cleverhans')

from cleverhans.attacks import FastGradientMethod, BasicIterativeMethod
from cleverhans.utils_tf import model_train, model_eval,model_loss
from cleverhans.model import Model

import time


def get_session():
    if tf.get_default_session() is None:
        print("Creating new session")
        tf.reset_default_graph()
        _SESSION = tf.InteractiveSession()
    else:
        print("Using old session")
        _SESSION = tf.get_default_session()

    return _SESSION


def get_gan_labels(lbls):
    # add class 0 which is the "fake" class
    if lbls is not None:
        labels = np.zeros((lbls.shape[0], lbls.shape[1] + 1))
        labels[:, 1:] = lbls
    else:
        labels = None
    return labels


def get_supervised_batches(dataset, size, batch_size, class_ids):

    def batchify_with_size(sampled_imgs, sampled_labels, size):
        rand_idx = np.random.choice(list(range(sampled_imgs.shape[0])), size, replace=False)
        imgs_ = sampled_imgs[rand_idx]
        lbls_ = sampled_labels[rand_idx]
        rand_idx = np.random.choice(list(range(imgs_.shape[0])), batch_size, replace=True)
        imgs_ = imgs_[rand_idx]
        lbls_ = lbls_[rand_idx] 
        return imgs_, lbls_

    labeled_image_batches, lblss = [], []
    num_passes = int(ceil(float(size) / batch_size))
    for _ in range(num_passes):
        for class_id in class_ids:
            labeled_image_batch, lbls = dataset.next_batch(int(ceil(float(batch_size)/len(class_ids))),
                                                           class_id=class_id)
            labeled_image_batches.append(labeled_image_batch)
            lblss.append(lbls)

    labeled_image_batches = np.concatenate(labeled_image_batches)
    lblss = np.concatenate(lblss)

    if size < batch_size:
        labeled_image_batches, lblss = batchify_with_size(labeled_image_batches, lblss, size)

    shuffle_idx = np.arange(lblss.shape[0]); np.random.shuffle(shuffle_idx)
    labeled_image_batches = labeled_image_batches[shuffle_idx]
    lblss = lblss[shuffle_idx]

    while True:
        i = np.random.randint(max(1, size/batch_size))
        yield (labeled_image_batches[i*batch_size:(i+1)*batch_size],
               lblss[i*batch_size:(i+1)*batch_size])


def get_test_batches(dataset, batch_size):

    try:
        test_imgs, test_lbls = dataset.test_imgs, dataset.test_labels
    except:
        test_imgs, test_lbls = dataset.get_test_set()

    all_test_img_batches, all_test_lbls = [], []
    test_size = test_imgs.shape[0]
    i = 0
    while (i+1)*batch_size <= test_size:
        all_test_img_batches.append(test_imgs[i*batch_size:(i+1)*batch_size])
        all_test_lbls.append(test_lbls[i*batch_size:(i+1)*batch_size])
        i += 1

    return all_test_img_batches, all_test_lbls



def get_test_accuracy(session, dcgan, all_test_img_batches, all_test_lbls):

    # only need this function because bdcgan has a fixed batch size for *everything*
    # test_size is in number of batches
    all_d_logits, all_s_logits, all_d = [], [], []
    for test_image_batch, test_lbls in zip(all_test_img_batches, all_test_lbls):
        test_d_logits, test_s_logits, test_d = session.run([dcgan.test_D_logits, dcgan.test_S_logits, dcgan.test_D],
                                                   feed_dict={dcgan.test_inputs: test_image_batch})
        all_d_logits.append(test_d_logits)
        all_s_logits.append(test_s_logits)
        all_d.append(test_d)

    test_d_logits = np.concatenate(all_d_logits)
    test_s_logits = np.concatenate(all_s_logits)
    test_lbls = np.concatenate(all_test_lbls)
    test_d = np.concatenate(all_d)
    
    #not_fake = np.where(np.argmax(test_d_logits, 1) > 0)[0]
    #not_fake = [i for i,x in enumerate(test_d) if x[0] < .5] #remove filter
    not_fake = np.where(test_d[:,0] < 0.5)[0] #remove filter
    if len(not_fake) < 10:
        print("WARNING: not enough samples for SS results")
    non_adv_acc = len(not_fake)/len(test_lbls)
    print("Test images discriminator thinks are not fake:" + str(len(not_fake)))
    semi_sup_acc = (100. * np.sum(np.argmax(test_d_logits[not_fake], 1) == np.argmax(test_lbls[not_fake], 1) + 1))\
                   / len(not_fake)
    sup_acc = (100. * np.sum(np.argmax(test_s_logits, 1) == np.argmax(test_lbls, 1)))\
              / test_lbls.shape[0]

    ex_prob = test_d[0]
    return sup_acc, semi_sup_acc, non_adv_acc, ex_prob

def get_adv_test_accuracy(session, dcgan, all_test_img_batches, all_test_lbls):

    # only need this function because bdcgan has a fixed batch size for *everything*
    # test_size is in number of batches
    all_d_logits, all_s_logits, all_d = [], [], []
    for test_image_batch, test_lbls in zip(all_test_img_batches, all_test_lbls):
        test_d_logits, test_d = session.run([dcgan.test_D_logits, dcgan.test_D],
                                                   feed_dict={dcgan.test_inputs: test_image_batch})

        all_d_logits.append(test_d_logits)
        all_d.append(test_d)

    test_d_logits = np.concatenate(all_d_logits)
    test_lbls = np.concatenate(all_test_lbls)
    test_d = np.concatenate(all_d)

    #not_fake = [i for i,x in enumerate(test_d) if x[0] < .5] #remove filter
    not_fake = np.where(test_d[:,0] < 0.5)[0] #remove filter
    #not_fake = np.where(np.argmax(test_d_logits, 1) > 0)[0]
    if len(not_fake) < 10:
        print("WARNING: not enough samples for SS results")
    print("Adversarial images discriminator thinks are not fake:" + str(len(not_fake)))
    adv_accuracy = len(not_fake)/len(test_lbls)
    semi_sup_acc = (100. * np.sum(np.argmax(test_d_logits[not_fake], 1) == np.argmax(test_lbls[not_fake], 1) + 1))\
                   / len(not_fake)
    semi_sup_acc_unfilter = (100. * np.sum(np.argmax(test_d_logits[:,1:], 1) == np.argmax(test_lbls, 1)))\
                   / len(test_d_logits)

    correct_certainty = np.mean([i[0] for i,j in zip(test_d,test_lbls) if np.argmax(i[1:]) == np.argmax(j)])
    uncorrect_certainty = np.mean([i[0] for i,j in zip(test_d,test_lbls) if np.argmax(i[1:]) != np.argmax(j)])

    adv_ex_prob = test_d[0]
    return semi_sup_acc, semi_sup_acc_unfilter, correct_certainty, uncorrect_certainty, adv_accuracy, adv_ex_prob

def get_certainty_for_adv(session, dcgan, all_test_img_batches,all_test_lbls):

    all_d = []
    for test_image_batch, test_lbls in zip(all_test_img_batches, all_test_lbls):
        test_d = session.run(dcgan.test_D,
                                                   feed_dict={dcgan.test_inputs: test_image_batch})
        all_d.append(test_d)

    test_d = np.concatenate(all_d)
    test_lbls = np.concatenate(all_test_lbls)

    correct_certainty = np.mean([i[0] for i,j in zip(test_d,test_lbls) if np.argmax(i[1:]) == np.argmax(j)])
    uncorrect_certainty = np.mean([i[0] for i,j in zip(test_d,test_lbls) if np.argmax(i[1:]) != np.argmax(j)])
    return correct_certainty, uncorrect_certainty


    


def b_dcgan(dataset, args):

    z_dim = args.z_dim
    x_dim = dataset.x_dim
    batch_size = args.batch_size
    dataset_size = dataset.dataset_size

    session = get_session()

    test_x = tf.placeholder(tf.float32, shape=(batch_size, 28, 28, 1))
    x = tf.placeholder(tf.float32, shape=(batch_size, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(batch_size, 10))

    unlabeled_batch_ph= tf.placeholder(tf.float32, shape=(batch_size, 28, 28, 1))
    labeled_image_ph = tf.placeholder(tf.float32, shape=(batch_size, 28, 28, 1))
    if args.random_seed is not None:
	    tf.set_random_seed(args.random_seed)
    # due to how much the TF code sucks all functions take fixed batch_size at all times
    dcgan = BDCGAN(x_dim, z_dim, dataset_size, batch_size=batch_size, J=args.J, M=args.M, 
                   lr=args.lr, optimizer=args.optimizer,
                   gen_observed=args.gen_observed, adv_train=args.adv_train,
                   num_classes=dataset.num_classes if args.semi_supervised else 1)
    if args.adv_test and args.semi_supervised:
        if args.basic_iterative:
            fgsm = BasicIterativeMethod(dcgan, sess=session)
            dcgan.adv_constructor = fgsm
            fgsm_params = {'eps': args.eps,
                    'eps_iter': float(args.eps/4),
                    'nb_iter': 4,
                    'ord': np.inf,
                   'clip_min': 0.,
                   'clip_max': 1.}
                   #,'y_target': None} 
        else:
            fgsm = FastGradientMethod(dcgan, sess=session)
            dcgan.adv_constructor = fgsm
            eval_params = {'batch_size': batch_size}
            fgsm_params = {'eps': args.eps,
                       'clip_min': 0.,
                       'clip_max': 1.}
        adv_x = fgsm.generate(x,**fgsm_params)
        adv_test_x = fgsm.generate(test_x,**fgsm_params)
        preds = dcgan.get_probs(adv_x)
    if args.adv_train:
        unlabeled_targets = np.zeros([batch_size,dcgan.K+1])
        unlabeled_targets[:,0] = 1
        fgsm_targeted_params = {'eps': args.eps,
                   'clip_min': 0.,
                   'clip_max': 1.,
                    'y_target': unlabeled_targets
                   }

    saver = tf.train.Saver()

    print("Starting session")
    session.run(tf.global_variables_initializer())

    prev_iters = 0
    if args.load_chkpt:
        saver.restore(session, args.chkpt)
        # Assume checkpoint is of the form "model_300"
        prev_iters = int(args.chkpt.split('/')[-1].split('_')[1])
        print("Model restored from iteration:", prev_iters)

    print("Starting training loop")
    num_train_iter = args.train_iter

    if hasattr(dataset, "supervised_batches"):
        # implement own data feeder if data doesnt fit in memory
        supervised_batches = dataset.supervised_batches(args.N, batch_size)
    else:
        supervised_batches = get_supervised_batches(dataset, args.N, batch_size, list(range(dataset.num_classes)))

    if args.semi_supervised:
        test_image_batches, test_label_batches = get_test_batches(dataset, batch_size)


        optimizer_dict = {"semi_d": dcgan.d_optim_semi_adam,
                          "sup_d": dcgan.s_optim_adam,
                          "adv_d": dcgan.d_optim_adam,
                          "gen": dcgan.g_optims_adam}
    else:
        optimizer_dict = {"adv_d": dcgan.d_optim_adam,
                          "gen": dcgan.g_optims_adam}

    base_learning_rate = args.lr # for now we use same learning rate for Ds and Gs
    lr_decay_rate = args.lr_decay

    for train_iter in range(1+prev_iters, 1+num_train_iter):

        if train_iter == 5000:
            print("Switching to user-specified optimizer")
            if args.semi_supervised:
                optimizer_dict = {"semi_d": dcgan.d_optim_semi,
                                  "sup_d": dcgan.s_optim,
                                  "adv_d": dcgan.d_optim,
                                  "gen": dcgan.g_optims}
            else:
                optimizer_dict = {"adv_d": dcgan.d_optim,
                                  "gen": dcgan.g_optims}

        learning_rate = base_learning_rate * np.exp(-lr_decay_rate *
                                                    min(1.0, (train_iter*batch_size)/float(dataset_size)))

        batch_z = np.random.uniform(-1, 1, [batch_size, z_dim])
        image_batch, batch_label = dataset.next_batch(batch_size, class_id=None)
        batch_targets = np.zeros([batch_size,11])
        batch_targets[:,0] = 1


        if args.semi_supervised:
            labeled_image_batch, labels = next(supervised_batches)
            if args.adv_train:
                adv_labeled = session.run(fgsm.generate(labeled_image_ph,**fgsm_targeted_params), feed_dict = {labeled_image_ph:labeled_image_batch})
                adv_unlabeled = session.run(fgsm.generate(unlabeled_batch_ph,**fgsm_params),feed_dict = {unlabeled_batch_ph:image_batch})
                _, d_loss = session.run([optimizer_dict["semi_d"], dcgan.d_loss_semi], feed_dict={dcgan.labeled_inputs: labeled_image_batch,
                                                                                                  dcgan.labels: get_gan_labels(labels),
                                                                                                  dcgan.inputs: image_batch,
                                                                                                  dcgan.z: batch_z,
                                                                                                  dcgan.d_semi_learning_rate: learning_rate,
                                                                                                  dcgan.adv_unlab: adv_unlabeled,
                                                                                                  dcgan.adv_labeled: adv_labeled
                                                                                                  })
            else:
                _, d_loss = session.run([optimizer_dict["semi_d"], dcgan.d_loss_semi], feed_dict={dcgan.labeled_inputs: labeled_image_batch,
                                                                                                  dcgan.labels: get_gan_labels(labels),
                                                                                                  dcgan.inputs: image_batch,
                                                                                                  dcgan.z: batch_z,
                                                                                                  dcgan.d_semi_learning_rate: learning_rate
                                                                                                  })

            _, s_loss = session.run([optimizer_dict["sup_d"], dcgan.s_loss], feed_dict={dcgan.inputs: labeled_image_batch,
                                                                                        dcgan.lbls: labels})
            
        else:
            # regular GAN
            _, d_loss = session.run([optimizer_dict["adv_d"], dcgan.d_loss], feed_dict={dcgan.inputs: image_batch,
                                                                                        dcgan.z: batch_z,
                                                                                        dcgan.d_learning_rate: learning_rate})


        if args.wasserstein:
            session.run(dcgan.clip_d, feed_dict={})

        g_losses = []
        for gi in range(dcgan.num_gen):

            # compute g_sample loss
            batch_z = np.random.uniform(-1, 1, [batch_size, z_dim])
            for m in range(dcgan.num_mcmc):
                _, g_loss = session.run([optimizer_dict["gen"][gi*dcgan.num_mcmc+m], dcgan.generation["g_losses"][gi*dcgan.num_mcmc+m]],
                                        feed_dict={dcgan.z: batch_z, dcgan.g_learning_rate: learning_rate})
                g_losses.append(g_loss)

        # if args.adv_test:
        #     probs, logits = dcgan.discriminator(adv_x,dcgan.K+1,reuse = True)
            
        #     labels = tf.placeholder(tf.float32,
        #                              [args.batch_size, dcgan.K+1], name='real_targets')
        #     compare_labels = tf.convert_to_tensor(np.array([np.append(0,i) for i in batch_label]))

        #     print(session.run(model_loss(compare_labels,probs), feed_dict = {x:image_batch}))
        # if args.adv_test:
        #     #preds = dcgan.get_probs(adv_x)
        #     #eval_preds = session.run(preds, feed_dict = {x:image_batch})
        #     #print(eval_preds[0])
        #     #adv_exs = session.run(adv_test_x, feed_dict = {x:test_image_batches})
        #     # adv_acc = model_eval(
        #     #     session, x, y, preds, image_batch, batch_label, args=eval_params)
        #     # #print(session.run(model_loss(compare_labels,probs), feed_dict = {x:image_batch}))
        #     # print("Adversarial loss = %2.f" % (1-adv_acc))
        #     print(get_test_accuracy(session,dcgan,adv_set,test_label_batches))

        if train_iter > 0 and train_iter % args.n_save == 0:
            print("Iter %i" % train_iter)
            # collect samples
            if args.save_samples: # saving samples
                all_sampled_imgs = []
                for gi in range(dcgan.num_gen):
                    _imgs, _ps = [], []
                    for _ in range(10):
                        sample_z = np.random.uniform(-1, 1, size=(batch_size, z_dim))
                        sampled_imgs, sampled_probs = session.run([dcgan.generation["gen_samplers"][gi*dcgan.num_mcmc],
                                                                   dcgan.generation["d_probs"][gi*dcgan.num_mcmc]],
                                                                  feed_dict={dcgan.z: sample_z})
                        _imgs.append(sampled_imgs)
                        _ps.append(sampled_probs)

                    sampled_imgs = np.concatenate(_imgs); sampled_probs = np.concatenate(_ps)
                    all_sampled_imgs.append([sampled_imgs, sampled_probs[:, 1:].sum(1)])

            print("Disc loss = %.2f, Gen loss = %s" % (d_loss, ", ".join(["%.2f" % gl for gl in g_losses])))

            #if args.adv_test:
            #preds = dcgan.get_probs(adv_x)
            #eval_preds = session.run(preds, feed_dict = {x:image_batch})
            #print(eval_preds[0])
            #adv_exs = session.run(adv_test_x, feed_dict = {x:test_image_batches})
            # adv_acc = model_eval(
            #     session, x, y, preds, image_batch, batch_label, args=eval_params)
            # #print(session.run(model_loss(compare_labels,probs), feed_dict = {x:image_batch}))
            # print("Adversarial loss = %2.f" % (1-adv_acc))
                #print(get_test_accuracy(session,dcgan,adv_set,test_label_batches))

                # adv_x = fgsm.generate(x,**fgsm_params)
                # preds = dcgan.get_probs(adv_x)
                # acc = model_eval(
                #     session, x, y, preds, image_batch, batch_label, args=eval_params)
                # print("Adversarial loss = %2.f" % (1-acc))


            if args.semi_supervised:
                # get test set performance on real labels only for both GAN-based classifier and standard one

                s_acc, ss_acc, non_adv_acc, ex_prob = get_test_accuracy(session, dcgan, test_image_batches, test_label_batches)
                if args.adv_test:
                    adv_set = []
                    for test_images in test_image_batches:
                        adv_set.append(session.run(adv_x, feed_dict = {x:test_images}))
                    adv_sup_acc, adv_ss_acc,correct_uncertainty, incorrect_uncertainty, adv_acc, adv_ex_prob = get_adv_test_accuracy(session,dcgan,adv_set,test_label_batches)
                    print("Adversarial semi-sup accuracy with filter: %.2f" % adv_sup_acc)
                    print("Adverarial semi-sup accuracy: %.2f" % adv_ss_acc)
                    print("Uncertainty for correct predictions: %.2f" % correct_uncertainty)
                    print("Uncertainty for incorrect predictions: %.2f" % incorrect_uncertainty)
                    print("non_adversarial_classification_accuracy: %.2f" % non_adv_acc)
                    print("adversarial_classification_accuracy: %.2f" % adv_acc)
                    
                    if args.save_samples:
                        print("saving adversarial test images and test images")
                        i = 0

                        for x,y in zip(adv_set[-1], test_image_batches[-1]):
                            np.save(args.out_dir+'/adv_test'+str(train_iter)+'_'+str(i), x)
                            np.save(args.out_dir+'/test'+str(train_iter)+'_'+str(i), y)
                            i = i+1
                            if i==5: #save 5 adversarial images 
                                break
                   


                print("Supervised acc: %.2f" % (s_acc))
                print("Semi-sup acc: %.2f" % (ss_acc))

            print("saving results and samples")

            results = {"disc_loss": float(d_loss),
                       "gen_losses": list(map(float, g_losses))}
            if args.semi_supervised:
                #results["example_non_adversarial_probs"] = list(ex_prob.flatten())
                #results["example_adversarial_probs"] = list(adv_ex_prob.flatten())
                results["non_adversarial_classification_accuracy"] = float(non_adv_acc)
                results["adversarial_classification_accuracy"] = float(adv_acc)
                results["adversarial_uncertainty_correct"] = float(correct_uncertainty)
                results["adversarial_uncertainty_incorrect"] = float(incorrect_uncertainty)
                results["supervised_acc"] = float(s_acc)
                results['adversarial_filtered_semi_supervised_acc'] = float(adv_sup_acc)
                results["adversarial_unfilted_semi_supervised_acc"] = float(adv_ss_acc)
                results["semi_supervised_acc"] = float(ss_acc)
                results["timestamp"] = time.time()
                results["previous_chkpt"] = args.chkpt

            with open(os.path.join(args.out_dir, 'results_%i.json' % train_iter), 'w') as fp:
                json.dump(results, fp)
            
            if args.save_samples:
                for gi in range(dcgan.num_gen):
                    print_images(all_sampled_imgs[gi], "B_DCGAN_%i_%.2f" % (gi, g_losses[gi*dcgan.num_mcmc]),
                                 train_iter, directory=args.out_dir)

                print_images(image_batch, "RAW", train_iter, directory=args.out_dir)


            if args.save_weights:
                var_dict = {}
                for var in tf.trainable_variables():
                    var_dict[var.name] = session.run(var.name)

                np.savez_compressed(os.path.join(args.out_dir,
                                                 "weights_%i.npz" % train_iter),
                                    **var_dict)
            

            print("Done saving weights")

        if train_iter > 0 and train_iter%args.save_chkpt == 0:
            save_path = saver.save(session, os.path.join(args.out_dir,
                                                         "model_%i" % train_iter))
            print("Model checkpointed in file: %s" % save_path)
            
 
    session.close()    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Script to run Bayesian GAN experiments')

    parser.add_argument('--out_dir',
                        type=str,
                        required=True,
                        help="location of outputs (root location, which exists)")

    parser.add_argument('--n_save',
                        type=int,
                        default=100,
                        help="every n_save iteration save samples and weights")
    
    parser.add_argument('--z_dim',
                        type=int,
                        default=100,
                        help='dim of z for generator')
    
    parser.add_argument('--gen_observed',
                        type=int,
                        default=1000,
                        help='number of data "observed" by generator')

    parser.add_argument('--data_path',
                        type=str,
                        required=True,
                        help='path to where the datasets live')

    parser.add_argument('--dataset',
                        type=str,
                        default="mnist",
                        help='datasate name mnist etc.')

    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help="minibatch size")

    parser.add_argument('--prior_std',
                        type=float,
                        default=1.0,
                        help="NN weight prior std.")

    parser.add_argument('--numz',
                        type=int,
                        dest="J",
                        default=1,
                        help="number of samples of z to integrate it out")

    parser.add_argument('--num_mcmc',
                        type=int,
                        dest="M",
                        default=1,
                        help="number of MCMC NN weight samples per z")

    parser.add_argument('--eps',
                        type=int,
                        dest="eps",
                        default=.25,
                        help="epsilon for FGSM")

    parser.add_argument('--N',
                        type=int,
                        default=128,
                        help="number of supervised data samples")

    parser.add_argument('--semi_supervised',
                        action="store_true",
                        help="do semi-supervised learning")

    parser.add_argument('--train_iter',
                        type=int,
                        default=50000,
                        help="number of training iterations")

    parser.add_argument('--adv_test',
                        action="store_true",
                        help="do adv testing")

    parser.add_argument('--adv_train',
                        action="store_true",
                        help="do adv training")

    parser.add_argument('--basic_iterative',
                        action="store_true",
                        help="do basic iterative method of adversarial attack")

    parser.add_argument('--wasserstein',
                        action="store_true",
                        help="wasserstein GAN")

    parser.add_argument('--ml_ensemble',
                        type=int,
                        default=0,
                        help="if specified, an ensemble of --ml_ensemble ML DCGANs is trained")

    parser.add_argument('--save_samples',
                        action="store_true",
                        help="whether to save generated samples")

    parser.add_argument('--save_weights',
                        action="store_true",
                        help="whether to save weights")

    parser.add_argument('--save_chkpt',
                        type=int,
                        default=200,
                        help="number of iterations per checkpoint")

    parser.add_argument('--load_chkpt',
                        action="store_true",
                        help="whether to load from a checkpoint")

    parser.add_argument('--chkpt',
                        type=str,
                        default='',
                        help="name of checkpoint to load")

    parser.add_argument('--custom_experiment',
                        type=str,
                        default='',
                        help="custom name of experiment")

    parser.add_argument('--random_seed',
                        type=int,
                        default=None,
                        help="random seed")

    parser.add_argument('--lr',
                        type=float,
                        default=0.005,
                        help="learning rate")

    parser.add_argument('--lr_decay',
                        type=float,
                        default=3.0,
                        help="learning rate")

    parser.add_argument('--optimizer',
                        type=str,
                        default="sgd",
                        help="optimizer --- 'adam' or 'sgd'")

    
    args = parser.parse_args()
    
    if args.random_seed is not None:
#        np.random.seed(args.random_seed)
        np.random.seed(2222)
        tf.set_random_seed(args.random_seed)

    if not os.path.exists(args.out_dir):
        print("Creating %s" % args.out_dir)
        os.makedirs(args.out_dir)
    if args.custom_experiment != '':
        args.out_dir = os.path.join(args.out_dir, args.custom_experiment)
    else:
        args.out_dir = os.path.join(args.out_dir, "bgan_%s_%s" % (args.dataset, str(datetime.now())))

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    import pprint
    with open(os.path.join(args.out_dir, "hypers.txt"), "w") as hf:
        hf.write("Hyper settings:\n")
        hf.write("%s\n" % (pprint.pformat(args.__dict__)))
        
    celeb_path = os.path.join(args.data_path, "celebA")
    cifar_path = os.path.join(args.data_path, "cifar-10-batches-py")
    svhn_path = os.path.join(args.data_path, "svhn")
    mnist_path = os.path.join(args.data_path, "mnist") # can leave empty, data will self-populate
    imagenet_path = os.path.join(args.data_path, args.dataset)
    #imagenet_path = os.path.join(args.data_path, "imagenet")

    if args.dataset == "mnist":
        dataset = MnistDataset(mnist_path)
    elif args.dataset == "celeb":
        dataset = CelebDataset(celeb_path)
    elif args.dataset == "cifar":
        dataset = Cifar10(cifar_path)
    elif args.dataset == "svhn":
        dataset = SVHN(svhn_path)
    elif "imagenet" in args.dataset:
        num_classes = int(args.dataset.split("_")[-1])
        dataset = ImageNet(imagenet_path, num_classes)
    else:
        raise RuntimeError("invalid dataset %s" % args.dataset)
        
    ### main call
    if args.ml_ensemble:
        from ml_dcgan import ml_dcgan
        root = args.out_dir
        for ens in range(args.ml_ensemble):
            args.out_dir = os.path.join(root, "%i" % ens)
            if not os.path.exists(args.out_dir):
                os.makedirs(args.out_dir)
            ml_dcgan(dataset, args)
    else:
        b_dcgan(dataset, args)
