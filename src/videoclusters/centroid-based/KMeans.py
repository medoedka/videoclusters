from collections import namedtuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import linalg as LA


class KMeans:
    """
    Class for performing KMeans clustering algorithm.

    Attributes
    ----------
        num_clusters : int
            Number of clusters to be created.
        unit : str
            Unit to perform calculations on.
            'GPU' to use graphic processor,
            everything else or None to use central processor.
        cluster_convergence_step : float
            The minimal value of one-step loss alteration
            to finish the calculations.
        max_iters : int
            The maximum number of iterations if {'cluster_convergence_step'}
            wasn't reached.
    """

    def __init__(self,
                 num_clusters: int = 3,
                 unit: str = 'GPU',
                 cluster_convergence_step: float = 1e-4,
                 max_iters: int = 100):
        self.num_clusters = num_clusters
        self.unit = unit
        self.cluster_convergence_step = cluster_convergence_step
        self.max_iters = max_iters
        self.iteration = 1
        self.loss_dict = {}
        self.cuda_device = torch.device('cuda:0')
        self.is_cuda = None
        self.data = None
        self.minimums = None
        self.cluster_kernels = None

    def __initialization_of_clusters(self) -> torch.Tensor:
        """
        Randomly creates '{num_clusters}' unique cluster centres.

        Returns
        -------
            torch.Tensor
                Matrix of clustors coordinates.
        """
        cluster_coordinates = self.data[torch.randperm(len(self.data))[:self.num_clusters]]
        if self.is_cuda:
            cluster_coordinates = cluster_coordinates.to(device=self.cuda_device)
        return cluster_coordinates

    def __find_minimum(self) -> torch.Tensor:
        """
        Find minimal distances between data points and cluster centres.
        
        Returns
        -------
            minimums : torch.Tensor
                Vector, where each index represents datapoint index and
                value, corresponding to index, is the cluster index value.
                This is a compact way to relate every datapoint to its closest
                cluster.            
        """
        distances = torch.cdist(torch.unsqueeze(self.cluster_kernels, 0),
                                torch.unsqueeze(self.data.unsqueeze(1), 0))
        minimums = torch.flatten(torch.min(distances, dim=2).indices).long()
        return minimums

    def __count_loss(self, minimums: torch.Tensor) -> float:
        """
        Calculates the loss value for the one iteration.

        Parameters
        ----------
            minimums : torch.Tensor
                Vector, where each index represents datapoint index and
                value, corresponding to index, is the cluster index value.
                This is a compact way to relate every datapoint to its closest
                cluster.
        
        Returns
        -------
            float
                Loss value for the iteration.
        """
        minimums = torch.index_select(self.cluster_kernels, 0, minimums)
        loss = LA.norm(self.data - minimums, axis=1).mean().item()
        self.loss_dict[self.iteration] = loss
        return loss

    def __centroid_correction(self, minimums: torch.Tensor) -> torch.Tensor:
        """
        Recalculate culsters coordinates.

        Parameters
        ----------
            minimums : torch.Tensor
                Vector, where each index represents datapoint index and
                value, corresponding to index, is the cluster index value.
                This is a compact way to relate every datapoint to its closest
                cluster.    
        
        Returns
        -------
            torch.Tensor
                Matrix of clustors coordinates.
        """
        minimums = minimums.view(minimums.size(0), 1).expand(-1, self.data.size(1))
        unique_minimums, minimums_count = minimums.unique(dim=0, return_counts=True)
        cluster_coordinates = torch.zeros_like(unique_minimums, dtype=torch.float).scatter_add_(0, minimums, self.data)
        cluster_coordinates = cluster_coordinates / minimums_count.float().unsqueeze(1)
        return cluster_coordinates

    def fit_predict(self, data: torch.Tensor) -> namedtuple:
        """
        KMeans entrypoint function to calculate the clusters
        using giving data.

        Parameters
        ----------
            data : torch.Tensor
                Datapoints as a torch tensor.
        
        Returns
        -------
            namedtuple
                Namedtuple instance with major output info such as:
                iterations, loss, cluster centres and data labels.
        """
        if self.unit == 'GPU':
            if torch.cuda.is_available():
                self.is_cuda = True
                self.data = data.to(device=self.cuda_device)
        self.is_cuda = False
        self.data = data
        self.cluster_kernels = self.__initialization_of_clusters()
        while self.max_iters > 0:
            self.minimums = self.__find_minimum()
            loss = self.__count_loss(self.minimums)
            prev_iter = self.iteration - 1
            if prev_iter > 0 and abs(self.loss_dict[prev_iter] - loss) <= self.cluster_convergence_step:
                break
            self.cluster_kernels = self.__centroid_correction(self.minimums)
            self.iteration += 1
            self.max_iters -= 1

        result_names = namedtuple('Results', ['iterations',
                                              'loss',
                                              'cluster_centres',
                                              'data_labels'])
        result_values = result_names(self.iteration,
                                     list(self.loss_dict.values())[-1],
                                     self.cluster_kernels,
                                     self.minimums)
        return result_values

    def plot_losses(self) -> matplotlib.pyplot.Figure:
        """
        Draws the line plot of loss values at every iteration.

        Returns
        -------
            matplotlib.pyplot.Figure
                Line plot. X-axis is a number of iteration,
                Y-axis is the loss value.
        """
        x, y = zip(*self.loss_dict.items())
        plt.figure(figsize=(12, 10))
        plt.title('Change in loss against the number of iterations', fontsize=16)
        plt.xlabel('Iterations', fontsize=16)
        plt.ylabel('$J ^ {clust}$', fontsize=16)
        plt.xticks(np.arange(min(x), max(x) + 1, step=1))
        plt.plot(x, y, linestyle='-', marker='o', color='r')
        plt.show()

