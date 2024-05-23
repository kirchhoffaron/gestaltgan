import os
import sys
import re
from typing import List, Optional, Tuple, Union
import click
import dnnlib
import numpy as np
import torch
import cv2
from tqdm import tqdm

import legacy
import torchvision


def load_arcface(arcface_net: str, device: torch.tensor):
    net = torch.load(arcface_net)
    return net.eval().to(device)

def image_to_tensor(np_image: np.ndarray, device: torch.tensor):
    """
    Description
    ----
    Converts the numpy unit8 image to tensor
    """
    float_array = np_image / 127.5 - 1
    image_tensor = torch.tensor(
        float_array.transpose([2, 0, 1]),
        device=device
    )

    return image_tensor.unsqueeze(0).to(torch.float32).to(device)


def image_to_numpy(image_tensor: torch.Tensor):
    """
    Description
    ----
    Converts an image tensor to numpy unit8
    """
    n_w_h_c = image_tensor.permute(0, 2, 3, 1)
    uint8_array = (n_w_h_c * 127.5 + 128).clamp(0, 255).to(torch.uint8)

    return uint8_array.cpu().numpy()[0]



def generate_label(
    G: torch.nn.Module,
    device: torch.device,
    samples: int = 1,
    label: int = None
):
    """
    Description
    ----
    Takes an integer label and converts it to a one-hot encoded vector
    """
    labels = torch.zeros([samples, G.c_dim], device=device)
    if label:
        labels[:, label] = 1

    return labels


def sample_latent_vectors_w(G: torch.nn.Module, device: torch.device, samples: int, label: int=None):
    """
    Samples a latent vector in the W space
    """
    random_z = torch.rand(samples, G.z_dim).to(device)

    labels = generate_label(G, device, samples, label)

    return G.mapping(random_z, labels)

def generate_from_w(G: torch.nn.Module, latent_w: torch.tensor):
    w_repeat = latent_w.unsqueeze(0).repeat(1, G.mapping.num_ws, 1)
    return G.synthesis(w_repeat)


def perform_ganinversion(
        G: torch.nn.Module, 
        device: torch.device, 
        gm: torch.nn.Module, 
        image: np.ndarray, 
        steps: int=100, 
        label: int=None
    ) -> torch.tensor:
    """
    Searches the representation of an image in the latent space
    """
    
    arcface_input_dim = (112, 112)
    real_resized = cv2.resize(image, arcface_input_dim)
    real_features = gm(image_to_tensor(real_resized, device))[1]
    # Find a starting point for the ganinversion
    w_samples = sample_latent_vectors_w(G, device, 1000, label)
    w_samples = w_samples[:, :1, :]
    w_start = torch.mean(w_samples, axis=0, keepdims=True)

    w_optimize = torch.tensor(w_start, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([w_optimize], lr=0.1)
        
    resize = torchvision.transforms.Resize(arcface_input_dim)
    cos_similarity = torch.nn.CosineSimilarity()
    
    # Iteratively improve the latent vector
    for _ in range(steps):
        w_repeat = w_optimize.repeat(1, G.mapping.num_ws, 1)
        synth_image = G.synthesis(w_repeat, noise_mode='const')
    
        fake_resized = resize(synth_image)
        fake_features = gm(fake_resized)[1]
        loss = torch.absolute(1 - cos_similarity(real_features, fake_features)[0])
    
        optimizer.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        optimizer.step()

    return w_optimize



def perform_ganinversion_ensemble(
        G: torch.nn.Module, 
        device: torch.device, 
        gm1: torch.nn.Module, 
        gm2: torch.nn.Module, 
        arcface: torch.nn.Module, 
        image: np.ndarray, 
        steps: int=100, 
        label: int=None
    ) -> torch.tensor:
    """
    Searches the representation of an image in the latent space using the ensemble loss
    """
    arcface_input_dim = (112, 112)
    real_resized = cv2.resize(image, arcface_input_dim)
    real_features1 = gm1(image_to_tensor(real_resized, device))[1]
    real_features2 = gm2(image_to_tensor(real_resized, device))[1]
    real_features3 = arcface(image_to_tensor(real_resized, device))

    real_average = torch.stack((real_features1, real_features2, real_features3)).mean(axis=0)

    # Find a starting point for the ganinversion
    w_samples = sample_latent_vectors_w(G, device, 1000, label)
    w_samples = w_samples[:, :1, :]
    w_start = torch.mean(w_samples, axis=0, keepdims=True)

    w_optimize = torch.tensor(w_start, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([w_optimize], lr=0.1)
        
    resize = torchvision.transforms.Resize(arcface_input_dim)
    cos_similarity = torch.nn.CosineSimilarity()
    for _ in range(steps):
        w_repeat = w_optimize.repeat(1, G.mapping.num_ws, 1)
        synth_image = G.synthesis(w_repeat, noise_mode='const')
    
        fake_resized = resize(synth_image)
        fake_features1 = gm1(fake_resized)[1]
        fake_features2 = gm2(fake_resized)[1]
        fake_features3 = arcface(fake_resized)

        fake_average = torch.stack((fake_features1, fake_features2, fake_features3)).mean(axis=0)

        loss = torch.absolute(1 - cos_similarity(real_average, fake_average)[0])
    
        optimizer.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        optimizer.step()

    return w_optimize


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
    if class_idx is None and ganinversion is None:
        raise click.ClickException('Either --class or --ganinversion and gm1 has to be given')


    print('Loading network from "%s"...' % network)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    if ganinversion is not None and gm1 is not None:
        latent_vectors = []

        # load gm model 2 and 3
        if gm2 is not None and arcface is not None:
            print("Performing GANinversion with nets %s, %s, %s" % (gm1, gm2, arcface))
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
