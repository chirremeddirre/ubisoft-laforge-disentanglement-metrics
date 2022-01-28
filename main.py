#!/usr/bin/env python

import argparse
import models
import model_utils
import torch

def main(args):
    dataloader = model_utils.dsprites_loader(args.load_path, args.batch_size)
    if args.train:
        print(f"In train in the membrain: {args.train}")
        if args.model.lower() == "beta-vae":
                model = models.beta_VAE(args.image_size, args.latents).to(args.device)
                model, x_hat, x = model._train(args.epochs, dataloader, args.device)
        elif args.model.lower() == "pmp":
                print("Not yet implemented")
        else:
                print(f"Incorrect model type provided: {args.model}")
    else:
        model = models.beta_VAE(64,10)
        model.load_state_dict(torch.load(args.model_path, map_location=args.device))
        model.eval()
    if args.eval:
        dataset = model_utils.load_dsprites(args.load_path)
        model_utils.eval(model, args.model, dataset, args.metrics, args.device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Disentanglement')
    parser.add_argument('--latents', default=10, type=int, help='Number of latents')
    parser.add_argument('--epochs', default=1, type=int, help='Number of epochs used in training')
    parser.add_argument('--model', default="beta-vae", type=str, help="The type of model, e.g, beta-vae or PMP")
    parser.add_argument('--load_path', default="../dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz", type=str, help="Path to the dataset")
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--image_size', default=64, type=int, help='Size of the image(assumed to be square)')
    parser.add_argument('--device', default='cpu', type=str, help='Type of device cpu/cuda')
    parser.add_argument('--train', default=False, type=bool, help='Train a new model')
    parser.add_argument('--save_path', default="models", type=str, help="Path to save the newly trained model")
    parser.add_argument('--model_path', default="models", type=str, help="Path to a previously trained model")
    parser.add_argument('--eval', default=True, type=bool, help='Evaluate the model using some dissentanglement metric')
    parser.add_argument('--metrics', default="all", type=str, help='Dissentanglement metrics to use in evaluation, see documentation for valid metrics')
    args = parser.parse_args()
    main(args)
