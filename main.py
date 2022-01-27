#!/usr/bin/env python

import argparse
import models
import model_utils

def main(args):
    if args.train:
        if args.model.lower() == "beta-vae":
                model = models.beta_VAE(args.image_size, args.latents)
                dataset = model_utils.load_dsprites(args.path, args.batch_size)
                net, x_hat, x = model.train(args.epochs, dataset, args.device)
        elif args.model.lower() == "pmp":
                print("Not yet implemented")
        else:
                print(f"Incorrect model type provided: {args.model}")

    if args.eval:
        model_utils.eval(net, args.model, dataset, args.metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Disentanglement')
    parser.add_argument('--latents', default=10, type=int, help='Number of latents')
    parser.add_argument('--epochs', default=1, type=int, help='Number of epochs used in training')
    parser.add_argument('--model', default="beta-vae", type=str, help="The type of model, e.g, beta-vae or PMP")
    parser.add_argument('--load_path', default="../dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz", type=str, help="Path to the dataset")
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--image_size', default=64, type=int, help='Size of the image(assumed to be square)')
    parser.add_argument('--device', default='cpu', type=str, help='Type of device cpu/cuda')
    parser.add_argument('--train', default=False, type=str2bool, help='Train a new model')
    parser.add_argument('--save_path', default="models", type=str, help="Path to save the newly trained model")
    parser.add_argument('--eval', default=True, type=str2bool, help='Evaluate the model using some dissentanglement metric')
    parser.add_argument('--metrics', default="all", nargs='+', help='Dissentanglement metrics to use in evaluation, see documentation for valid metrics')
    args = parser.parse_args()
    main(args)
