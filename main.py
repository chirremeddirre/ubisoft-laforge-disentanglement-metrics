#!/usr/bin/env python
#
import argparse
import models
import model_utils
import torch

def main(args):
    if args.color:
        dataloader = model_utils.dsprites_color_loader(args.load_path, args.batch_size)
        args.image_chans = 3
    else:
        dataloader = model_utils.dsprites_loader(args.load_path, args.batch_size)
    if args.train:
        print(f"In train in the membrain: {args.train}")
        if args.model.lower() == "beta-vae":
                model = models.beta_VAE(args.image_size, args.image_chans, args.latents, args.beta).to(args.device)
                model, x_hat, x = model._train(args.epochs, dataloader, args.device)
        elif args.model.lower() == "pmp":
                print("Not yet implemented")
        else:
                print(f"Incorrect model type provided: {args.model}")
    else:
        model = model_utils.load_model(args.model, args.image_size, args.latents, args.model_path, args.device)
    if args.eval:
        dataset = model_utils.load_dsprites(args.load_path)
        model_utils.eval(model, args.model, dataset, args.metrics, args.batch_eval_size, args.eval_iters, args.device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Disentanglement')
    parser.add_argument('--latents', default=10, type=int, help='Number of latents')
    parser.add_argument('--epochs', default=1, type=int, help='Number of epochs used in training')
    parser.add_argument('--model', default="beta-vae", type=str, help="The type of model, e.g, beta-vae or PMP")
    parser.add_argument('--beta', default=4, type=int, help="Beta value to be used if training new beta-VAE model")
    parser.add_argument('--load_path', default="../dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz", type=str, help="Path to the dataset")
    parser.add_argument('--color', default=False, type=bool, help="Color Dsprites loader") # True will use a loader that which uniformly random assigns each shape a color (R,G,B)
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--image_size', default=64, type=int, help='Size of the image(assumed to be square)')
    parser.add_argument('--image_chans', default=1, type=int, help='Number of channels to the image, i.e, 3 if RGB, 1 if greyscale')
    parser.add_argument('--device', default='cpu', type=str, help='Type of device cpu/cuda')
    parser.add_argument('--train', default=False, type=bool, help='Train a new model')
    parser.add_argument('--save_path', default="models", type=str, help="Path to save the newly trained model")
    parser.add_argument('--model_path', default="models", type=str, help="Path to a previously trained model")
    parser.add_argument('--eval', default=True, type=bool, help='Evaluate the model using some dissentanglement metric')
    parser.add_argument('--metrics', default="all", type=str, help='Dissentanglement metrics to use in evaluation, see documentation for valid metrics')
    parser.add_argument('--batch_eval_size', default=200, type=int, help='Batch size for evaluation')
    parser.add_argument('--eval_iters', default=25, type=int, help='Number of iterations used in evaluation. Total samples in evaluation = batch_eval_size*eval_iters')
    args = parser.parse_args()
    main(args)
