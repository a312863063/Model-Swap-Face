# -*- coding:utf-8 -*-
import os
import argparse
import pickle
from tqdm import tqdm
import PIL.Image
from PIL import ImageFilter
import numpy as np
import dnnlib.tflib as tflib
from encoder.generator_model import Generator
from encoder.perceptual_model import load_image
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

    # Perceptual model params
    parser.add_argument('--image_size', default=256, help='Size of images for perceptual model', type=int)
    parser.add_argument('--resnet_image_size', default=256, help='Size of images for the Resnet model', type=int)
    parser.add_argument('--load_resnet', default='networks/finetuned_resnet.h5', help='Model to load for ResNet approximation of dlatents')
    parser.add_argument('--use_preprocess_input', default=True, help='Call process_input() first before using feed forward net', type=str2bool, nargs='?', const=True)

    # Generator params
    parser.add_argument('--randomize_noise', default=False, help='Add noise to dlatents during optimization', type=str2bool, nargs='?', const=True)
    parser.add_argument('--tile_dlatents', default=False, help='Tile dlatents to use a single vector at each scale', type=str2bool, nargs='?', const=True)
    parser.add_argument('--clipping_threshold', default=2.0, help='Stochastic clipping of gradient values outside of this threshold', type=float)

    # Masking params
    parser.add_argument('--composite_blur', default=8, help='Size of blur filter to smoothly composite the images', type=int)

    args, other_args = parser.parse_known_args()

    # Initialize generator and encoder model
    tflib.init_tf()
    with open('networks/projector_'+projector_name+'.pkl', 'rb') as f:
        projector = pickle.load(f)
    generator = Generator(projector, args.batch_size, clipping_threshold=args.clipping_threshold, tiled_dlatent=args.tile_dlatents, model_res=args.model_res, randomize_noise=args.randomize_noise)
    if (args.dlatent_avg != ''):
        generator.set_dlatent_avg(np.load(args.dlatent_avg))
    print("    Loading ResNet Model...")
    ff_model = load_model(args.load_resnet)

    # Find the dlatent of the image
    dlatents = []
    imgs = []
    for image, mask in zip(images, masks):
        if (args.use_preprocess_input):
            dlatent = ff_model.predict(preprocess_input((load_image(image, image_size=args.resnet_image_size))))
        else:
            dlatent = ff_model.predict((load_image(image, image_size=args.resnet_image_size)))
        if dlatent is not None:
            generator.set_dlatents(dlatent)

        # Using Projector to generate images
        generator.set_dlatents(dlatent)
        generated_images = generator.generate_images()

        # Merge images with new face
        img_array = generated_images[0]
        ori_img = image
        width, height = ori_img.size
        mask = mask.resize((width, height))
        mask = mask.filter(ImageFilter.GaussianBlur(args.composite_blur))
        mask = np.array(mask) / 255
        mask = np.expand_dims(mask, axis=-1)
        img_array = mask * np.array(img_array) + (1.0 - mask) * np.array(ori_img)
        img_array = img_array.astype(np.uint8)
        img = PIL.Image.fromarray(img_array, 'RGB')

        imgs.append(img)
        dlatents.append(dlatent)

    return imgs, dlatents

