import os
import sys
import click
import dnnlib
import torch
import cv2
from tqdm import tqdm

import legacy

from ganinversion.utils import load_arcface, image_to_numpy, sample_latent_vectors_w, perform_ganinversion, perform_ganinversion_ensemble, generate_from_w

@click.command()
@click.option('--network', help='Network pickle filename', required=True)
@click.option('--class', 'class_idx', type=int, help='Class label of the average, None if not specified')
@click.option('--ganinversion', type=str, help='Folder of portraits to average in latent space')
@click.option('--gm1', type=str, help='GestaltMatcher-Arc model 1 to use for the comparisons')
@click.option('--gm2', type=str, help='GestaltMatcher-Arc model 2 to use for the comparisons (ensemble)')
@click.option('--arcface', type=str, help='Arcface model to use for the comparisons (ensemble)')
@click.option('--samples', type=int, help='Samples to calculate the mean/ steps for GANinversion', default=1000)
@click.option('--outfile', help='Filename for average image', type=str, required=True)
def generate_latent_average(
    network: str,
    class_idx: int,
    ganinversion: str,
    gm1: str,
    gm2: str,
    arcface: str,
    samples: int,
    outfile: str
):
    """
    Generate the average for a given class or a cohort of patient portraits
    """

    if class_idx is None and ganinversion is None:
        raise click.ClickException('Either --class or --ganinversion and gm1 has to be given')

    print('Loading network from "%s"...' % network)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    if ganinversion is not None and gm1 is not None:
        latent_vectors = []

        if gm2 is not None and arcface is not None:
            print("Performing GANinversion with nets %s, %s, %s" % (gm1, gm2, arcface))
            # load gm model 2 and 3
            gm2 = load_arcface(gm2, device)
            arcface = load_arcface(arcface, device)
        else:
            print("Performing GANinversion with net %s" % gm1)
        # load gm model 1
        gm1 = load_arcface(gm1, device)
        
        # perform ganinversion for all images in the cohort
        for file in tqdm(os.listdir(ganinversion)):
            image = cv2.imread(
                os.path.join(ganinversion, file)
            )
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if gm2 is not None and arcface is not None:
                latent_vectors += [perform_ganinversion_ensemble(G, device, gm1, gm2, arcface, image_rgb, samples, class_idx)]
            else:
                latent_vectors += [perform_ganinversion(G, device, gm1, image_rgb, samples, class_idx)]
        latent_vectors = torch.cat(latent_vectors)
    else:
        # sample random latent vectors for a given class
        latent_vectors = sample_latent_vectors_w(G, device, samples, label=class_idx)[:,0,:]

    # average the latent vectors
    latent_mean = torch.mean(latent_vectors, axis=0)
    latent_average = generate_from_w(G, latent_mean)

    latent_average_np = image_to_numpy(latent_average)
    image_rgb = cv2.cvtColor(latent_average_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(outfile, image_rgb)

    print('Latent average saved to %s' % outfile)

if __name__ == "__main__":
    generate_latent_average()
