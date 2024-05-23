import torch
import cv2
import torchvision


def load_arcface(arcface_net, device):
    net = torch.load(arcface_net)
    return net.eval().to(device)

def image_to_tensor(np_image, device):
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


def sample_latent_vectors_w(G, device, samples, label=None):
    random_z = torch.randn(samples, G.z_dim).to(device)

    labels = generate_label(G, device, samples, label)

    return G.mapping(random_z, labels)

def generate_from_w(G, latent_w):
    w_repeat = latent_w.unsqueeze(0).repeat(1, G.mapping.num_ws, 1)
    return G.synthesis(w_repeat)


def perform_ganinversion(G, device, arcface, image, steps=100, label=None):
    arcface_input_dim = (112, 112)
    real_resized = cv2.resize(image, arcface_input_dim)
    real_features = arcface(image_to_tensor(real_resized, device))[1]
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
        fake_features = arcface(fake_resized)[1]
        loss = torch.absolute(1 - cos_similarity(real_features, fake_features)[0])
    
        optimizer.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        optimizer.step()

    return w_optimize


def ensemble_loss(image_tensor, gm1, gm2, arcface):
    grayscale = torchvision.transforms.Grayscale(num_output_channels=3)
    features = []
    for flip in [False, True]:
        for gray in [False, True]:
            if flip:
                image_tensor = torchvision.transforms.functional.hflip(image_tensor)
            if gray:
                image_tensor = grayscale(image_tensor)

            features.append(gm1(image_tensor)[1])
            features.append(gm2(image_tensor)[1])
            features.append(arcface(image_tensor))

    return torch.stack(features).mean(axis=0)


def perform_ganinversion_ensemble(G, device, gm1, gm2, arcface, image, steps=100, label=None):
    arcface_input_dim = (112, 112)
    real_resized = cv2.resize(image, arcface_input_dim)
    real_tensor = image_to_tensor(real_resized, device)
    
    real_average = ensemble_loss(real_tensor, gm1, gm2, arcface)

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
        fake_average = ensemble_loss(fake_resized, gm1, gm2, arcface)

        loss = torch.absolute(1 - cos_similarity(real_average, fake_average)[0])
    
        optimizer.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        optimizer.step()

    return w_optimize

