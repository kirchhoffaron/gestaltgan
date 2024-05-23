import os

import click
import dnnlib
import numpy as np
import torch
import cv2
from tqdm import tqdm

import legacy

from ganinversion.utils import load_arcface, perform_ganinversion, perform_ganinversion_ensemble
import sys


@click.command()
@click.option('--network', help='Network pickle filename', required=True)
@click.option('--images', type=str, help='Folder of portraits to average in latent space', required=True)
@click.option('--gm1', type=str, help='GestaltMatcher-Arc model 1 to use for the comparisons', required=True)
@click.option('--gm2', type=str, help='GestaltMatcher-Arc model 2 to use for the comparisons (ensemble)')
@click.option('--arcface', type=str, help='Arcface model to use for the comparisons (ensemble)')
@click.option('--samples', type=int, help='Samples to calculate the mean/ steps for GANinversion', default=1000)
@click.option('--outfile', help='Filename for average image', type=str, required=True)
def generate_latent_average(
    network: str,
    images: str,
    gm1: str,
    gm2: str,
    arcface: str,
    samples: int,
    outfile: str
):
    print('Loading network from "%s"...' % network)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    latent_vectors = {}

    if gm2 is not None and arcface is not None:
        print("Performing GANinversion with nets %s, %s, %s" % (gm1, gm2, arcface))
        gm2 = load_arcface(gm2, device)
        arcface = load_arcface(arcface, device)
    else:
        print("Performing GANinversion with net %s" % gm1)
        
    gm1 = load_arcface(gm1, device)
        
    for file in tqdm(os.listdir(images)):
        image = cv2.imread(
            os.path.join(images, file)
        )
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if gm2 is not None and arcface is not None:
            latent_vectors[file] = perform_ganinversion_ensemble(G, device, gm1, gm2, arcface, image_rgb, samples, 0).detach().cpu().numpy()
        else:
            latent_vectors[file] = perform_ganinversion(G, device, gm1, image_rgb, samples, 0).detach().cpu().numpy()

    np.savez(outfile, **latent_vectors)

    print('Latent vectors saved to %s' % outfile)

if __name__ == "__main__":
    generate_latent_average()
