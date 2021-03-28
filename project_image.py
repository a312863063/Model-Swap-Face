# -*- coding:utf-8 -*-
import os
import argparse
import pickle
from tqdm import tqdm
import PIL.Image
from PIL import ImageFilter
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import tensorflow as tf
from encoder.generator_model import Generator
from encoder.perceptual_model import PerceptualModel, load_image
#from tensorflow.keras.models import load_model
from keras.models import load_model
from keras.applications.resnet50 import preprocess_input

def split_to_batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def project(images, masks, projector_name):
    parser = argparse.ArgumentParser(description='Find latent representation of reference images using perceptual losses', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dlatent_avg', default='', help='Use dlatent from file specified here for truncation instead of dlatent_avg from Gs')
    parser.add_argument('--model_res', default=1024, help='The dimension of images in the StyleGAN model', type=int)
    parser.add_argument('--batch_size', default=1, help='Batch size for generator and perceptual model', type=int)
    parser.add_argument('--optimizer', default='ggt', help='Optimization algorithm used for optimizing dlatents')

    # Perceptual model params
    parser.add_argument('--image_size', default=256, help='Size of images for perceptual model', type=int)
    parser.add_argument('--resnet_image_size', default=256, help='Size of images for the Resnet model', type=int)
    parser.add_argument('--lr', default=0.25, help='Learning rate for perceptual model', type=float)
    parser.add_argument('--decay_rate', default=0.9, help='Decay rate for learning rate', type=float)
    parser.add_argument('--iterations', default=1000, help='Number of optimization steps for each batch', type=int)
    parser.add_argument('--decay_steps', default=4, help='Decay steps for learning rate decay (as a percent of iterations)', type=float)
    parser.add_argument('--early_stopping', default=True, help='Stop early once training stabilizes', type=str2bool, nargs='?', const=True)
    parser.add_argument('--early_stopping_threshold', default=0.5, help='Stop after this threshold has been reached', type=float)
    parser.add_argument('--early_stopping_patience', default=10, help='Number of iterations to wait below threshold', type=int)
    parser.add_argument('--load_resnet', default='networks/finetuned_resnet.h5', help='Model to load for ResNet approximation of dlatents')
    parser.add_argument('--use_preprocess_input', default=True, help='Call process_input() first before using feed forward net', type=str2bool, nargs='?', const=True)
    parser.add_argument('--use_best_loss', default=True, help='Output the lowest loss value found as the solution', type=str2bool, nargs='?', const=True)
    parser.add_argument('--average_best_loss', default=0.25, help='Do a running weighted average with the previous best dlatents found', type=float)
    parser.add_argument('--sharpen_input', default=True, help='Sharpen the input images', type=str2bool, nargs='?', const=True)

    # Loss function options
    parser.add_argument('--use_vgg_loss', default=0.4, help='Use VGG perceptual loss; 0 to disable, > 0 to scale.', type=float)
    parser.add_argument('--use_vgg_layer', default=9, help='Pick which VGG layer to use.', type=int)
    parser.add_argument('--use_pixel_loss', default=1.5, help='Use logcosh image pixel loss; 0 to disable, > 0 to scale.', type=float)
    parser.add_argument('--use_mssim_loss', default=200, help='Use MS-SIM perceptual loss; 0 to disable, > 0 to scale.', type=float)
    parser.add_argument('--use_lpips_loss', default=100, help='Use LPIPS perceptual loss; 0 to disable, > 0 to scale.', type=float)
    parser.add_argument('--use_l1_penalty', default=0.5, help='Use L1 penalty on latents; 0 to disable, > 0 to scale.', type=float)
    parser.add_argument('--use_discriminator_loss', default=0.5, help='Use trained discriminator to evaluate realism.', type=float)
    parser.add_argument('--use_adaptive_loss', default=False, help='Use the adaptive robust loss function from Google Research for pixel and VGG feature loss.', type=str2bool, nargs='?', const=True)

    # Generator params
    parser.add_argument('--randomize_noise', default=False, help='Add noise to dlatents during optimization', type=str2bool, nargs='?', const=True)
    parser.add_argument('--tile_dlatents', default=False, help='Tile dlatents to use a single vector at each scale', type=str2bool, nargs='?', const=True)
    parser.add_argument('--clipping_threshold', default=2.0, help='Stochastic clipping of gradient values outside of this threshold', type=float)

    # Masking params
    parser.add_argument('--composite_blur', default=8, help='Size of blur filter to smoothly composite the images', type=int)

    args, other_args = parser.parse_known_args()
    args.decay_steps *= 0.01 * args.iterations  # Calculate steps as a percent of total iterations

    # Initialize generator and perceptual model
    tflib.init_tf()
    with open('networks/karras2019stylegan-ffhq-1024x1024.pkl','rb') as f:
        generator_network, discriminator_network, Gs_network = pickle.load(f)

    generator = Generator(Gs_network, args.batch_size, clipping_threshold=args.clipping_threshold, tiled_dlatent=args.tile_dlatents, model_res=args.model_res, randomize_noise=args.randomize_noise)
    if (args.dlatent_avg != ''):
        generator.set_dlatent_avg(np.load(args.dlatent_avg))

    perc_model = None
    if (args.use_lpips_loss > 0.00000001):
        with open('networks/vgg16_zhang_perceptual.pkl', 'rb') as f:
            perc_model = pickle.load(f)
    perceptual_model = PerceptualModel(args, perc_model=perc_model, batch_size=args.batch_size)
    perceptual_model.build_perceptual_model(generator, discriminator_network)

    ff_model = None

    # Optimize (only) dlatents by minimizing perceptual loss between reference and generated images in feature space
    best_dlatents = []
    for image, mask in zip(images, masks):
        perceptual_model.set_reference_image(image)
        dlatents = None
        if (ff_model is None):
            if os.path.exists(args.load_resnet):
                print("Loading ResNet Model:")
                ff_model = load_model(args.load_resnet)
        if (ff_model is not None): # predict initial dlatents with ResNet model
            if (args.use_preprocess_input):
                dlatents = ff_model.predict(preprocess_input(load_image(image, image_size=args.resnet_image_size)))
            else:
                dlatents = ff_model.predict(load_image(image,image_size=args.resnet_image_size))
        if dlatents is not None:
            generator.set_dlatents(dlatents)
        op = perceptual_model.optimize(generator.dlatent_variable, iterations=args.iterations, use_optimizer=args.optimizer)
        pbar = tqdm(op, leave=False, total=args.iterations)
        best_loss = None
        best_dlatent = None
        avg_loss_count = 0
        if args.early_stopping:
            avg_loss = prev_loss = None
        for loss_dict in pbar:
            if args.early_stopping:  # early stopping feature
                if prev_loss is not None:
                    if avg_loss is not None:
                        avg_loss = 0.5 * avg_loss + (prev_loss - loss_dict["loss"])
                        if avg_loss < args.early_stopping_threshold:  # count while under threshold; else reset
                            avg_loss_count += 1
                        else:
                            avg_loss_count = 0
                        if avg_loss_count > args.early_stopping_patience:  # stop once threshold is reached
                            break
                    else:
                        avg_loss = prev_loss - loss_dict["loss"]
            pbar.set_description(" Oprimizing dlatent: " + "; ".join(["{} {:.4f}".format(k, v) for k, v in loss_dict.items()]))
            if best_loss is None or loss_dict["loss"] < best_loss:
                if best_dlatent is None or args.average_best_loss <= 0.00000001:
                    best_dlatent = generator.get_dlatents()
                else:
                    best_dlatent = 0.25 * best_dlatent + 0.75 * generator.get_dlatents()
                if args.use_best_loss:
                    generator.set_dlatents(best_dlatent)
                best_loss = loss_dict["loss"]
            generator.stochastic_clip_dlatents()
            prev_loss = loss_dict["loss"]
        if not args.use_best_loss:
            best_loss = prev_loss
        best_dlatents.append(best_dlatent)
        print("\n Optimizing dlatent Best Loss {:.4f}".format(best_loss))

    # Using Projector to generate images
    tflib.init_tf()
    with open('networks/projector_'+projector_name+'.pkl', 'rb') as f:
        Gs_network = pickle.load(f)
    generator = Generator(Gs_network, args.batch_size, clipping_threshold=args.clipping_threshold,
                          tiled_dlatent=args.tile_dlatents, model_res=args.model_res,
                          randomize_noise=args.randomize_noise)
    imgs = []
    for best_dlatent, image, mask in zip(best_dlatents, images, masks):
        generator.set_dlatents(best_dlatent)
        img_array = generator.generate_images()[0]
        generator.reset_dlatents()

        # Merge images with new face
        width, height = image.size
        mask = mask.resize((width, height))
        mask = mask.filter(ImageFilter.GaussianBlur(args.composite_blur))
        mask = np.array(mask) / 255
        mask = np.expand_dims(mask, axis=-1)
        img_array = mask * np.array(img_array) + (1.0 - mask) * np.array(image)
        img_array = img_array.astype(np.uint8)
        img = PIL.Image.fromarray(img_array, 'RGB')
        imgs.append(img)

    return imgs, best_dlatents