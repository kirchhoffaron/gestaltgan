import torch
import torchvision
import sys
from onnx2torch import convert
import json
import numpy as np
from typing import Tuple

class GestaltLoss(torch.nn.Module):
    def __init__(
            self, 
            device: torch.device, 
            gm_a_path: str, 
            gm_b_path: str, 
            cfps_encodings: str, 
            loss_weighting:float=0.003
        ) -> None:
        super().__init__()
        self.device = device
        self.loss_weighting = loss_weighting

        self.load_models(gm_a_path, gm_b_path)
        self.encodings, self.labels = self.load_encodings(cfps_encodings)

        self.resize = torchvision.transforms.Resize(112)
        self.grayscale = torchvision.transforms.Grayscale(num_output_channels=3)
        self.cosine_similarity = torch.nn.CosineSimilarity()


    def load_encodings(self, cfps_encodings: str) -> Tuple[torch.tensor, torch.tensor]:
        """
        Load the encodings for the training images in the cfps
        """
        content = np.load(cfps_encodings, allow_pickle=True)
        cfps_vectors = content.flatten()[0]['vectors']
        labels = content.flatten()[0]['labels']
        
        labels = torch.tensor(labels).to(self.device)
        cfps_vectors = torch.tensor(cfps_vectors).to(self.device)

        return cfps_vectors, labels


    def load_model(self, path: str) -> torch.nn.Module:
        net = torch.load(path)
        return net.eval().to(self.device)


    def load_models(self, gm_a_path: str, gm_b_path: str):
        self.model_a = self.load_model(gm_a_path).eval()
        self.model_b = self.load_model(gm_b_path).eval()


    def preprocess(self, images: torch.tensor, flip: bool=False, gray: bool=False) -> torch.tensor:
        image = self.resize(images)

        if flip:
            images = torchvision.transforms.functional.hflip(images)
        if gray:
            images = self.grayscale(images)

        return image


    def ensemble(self, images):
        predictions = []
        
        for flip in [False, True]:
            for gray in [False, True]:
                images = self.preprocess(images, flip=flip, gray=gray)
                prediction_a = self.model_a(images)[1]
                prediction_b = self.model_b(images)[1]

                predictions += [prediction_a, prediction_b]

        return torch.stack(predictions).mean(axis=0)


    def forward(self, images: torch.tensor, class_labels: torch.tensor) -> torch.tensor:
        """
        Calculates the gestaltloss for a batch of images

        # expected shapes:
        # images => [b, 3, w, h]
        # class_labels => [b, c]
        """
        
        cfps_vectors = self.ensemble(images)

        batch_loss = 0

        for vector, label in zip(cfps_vectors, class_labels):
            label = torch.argmax(label)
            if label == 0:
                continue

            cosine_similarities = torch.absolute(
                1 - self.cosine_similarity(
                    vector.unsqueeze(dim=0),
                    self.encodings
                )
            )
            
            # calculate the index of the correct class in the GM prediction
            indexes = torch.argsort(cosine_similarities)
            labels = self.labels[indexes]           
            correct_index = torch.nonzero(labels == label)[0]
            loss = torch.numel(torch.unique(labels[:correct_index]))
            batch_loss += loss * self.loss_weighting

        return batch_loss