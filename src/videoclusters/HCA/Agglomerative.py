from typing import Tuple

import torch


class Agglomerative:
    """
    Agglomerative hierarchical clustering method.

    Attributes
    ----------
        num_clusters : int
            Number of clusters to be created.
        unit : str
            Unit to perform calculations on.
            'GPU' to use graphic processor,
            everything else or None to use central processor.
    """

    def __init__(self,
                 num_clusters: int = 2,
                 unit: str = 'GPU'):
        self.num_clusters = num_clusters
        self.unit = unit
        self.device = None
        self.data = None
        self.cluster_vector = None
        self.distance_matrix = None
    
    def get_distance_matrix(self, matrix : torch.Tensor) -> torch.Tensor:
        """
        Calculates distance matrix from given datapoints. In addition,
        upper triangle values are replaced by 'inf' values
        since they are just repeated lower triangle elements.

        Parameters
        ----------
            matrix : torch.Tensor
                Matrix, where each row represents the datapoint
                vector. Its number of rows is the number of datapoints.

        Returns
        -------
            torch.Tensor
                Distance matrix.
        """
        multiplied_matrixes = torch.mm(matrix, matrix.t())
        diag = multiplied_matrixes.diag().unsqueeze(0)
        diag = diag.expand_as(multiplied_matrixes)
        output_matrix = diag + diag.t() - 2 * multiplied_matrixes
        output_matrix = torch.tril(output_matrix, diagonal=0)
        output_matrix[output_matrix == 0.0] = float('inf')
        return output_matrix.sqrt()
    
    def __find_minimum_index(self) -> Tuple[int, int]:
        """
        Finds the index of the smallest value in the
        distance matrix.

        Returns
        -------
            Tuple[int, int]
                List [row, col], containing index location of the smallest value
                in the distance matrix.
        """
        index = torch.topk(self.distance_matrix.flatten(), k=1, largest=False).indices
        index = [index.item() // self.distance_matrix.size()[0],
                 index.item() % self.distance_matrix.size()[1]]
        return index
    
    def __get_cluster_elements(self,
                               element : int,
                               cluster_indexes : torch.Tensor) -> torch.Tensor:
        """
        Finds the elements of the cluster. 

        Parameters
        ----------
            element : int
                0 for the first cluster,
                1 for the second cluster.
            cluster_indexes : torch.Tensor
                Tensor, containing both cluster indexes.

        Returns
        -------
            torch.Tensor
                Elements of the cluster.
        """
        return (self.cluster_vector == cluster_indexes[element]).nonzero()
    
    def __merging_step(self):
        """
        Merges two clusters together at each iteration step.
        """
        min_index = self.__find_minimum_index()
        cluster_indexes = self.cluster_vector[min_index]
        cluster_1 = self.__get_cluster_elements(0, cluster_indexes)
        cluster_2 = self.__get_cluster_elements(1, cluster_indexes)

        # Update of the distance matrix by setting connected clusters values to 'inf'
        self.distance_matrix[cluster_1.flatten().repeat(1, len(cluster_2)).long(),
                             cluster_2.repeat_interleave(len(cluster_1)).long()] = float('inf')
        self.distance_matrix[cluster_2.flatten().repeat(1, len(cluster_1)).long(),
                             cluster_1.repeat_interleave(len(cluster_2)).long()] = float('inf')
        
        # Merging two clusters together
        self.cluster_vector = torch.where(self.cluster_vector == cluster_indexes[0],
                                          cluster_indexes[1],
                                          self.cluster_vector)

    def fit_predict(self, data : torch.Tensor) -> torch.Tensor:
        """
        Agglomerative entrypoint function to iteratively
        calculate the clusters using giving data.

        Parameters
        ----------
            data : torch.Tensor
                Datapoints as a torch tensor.

        Returns
        -------
            torch.Tensor
                Vector, where each index represents datapoint index and
                value, corresponding to index, is the cluster index value.
                This is a compact way to relate every datapoint to its closest
                cluster.
        """
        if self.unit == 'GPU':
            if torch.cuda.is_available():
                self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.data = data.to(self.device)
        self.num_initial_clusters = len(self.data)
        self.distance_matrix = self.get_distance_matrix(self.data).to(self.device)
        self.cluster_vector = torch.arange(0, len(self.data), 1).to(self.device)

        while self.num_initial_clusters > self.num_clusters:
            self.__merging_step()
            self.num_initial_clusters -= 1
        
        # change the indexes of the clusters to be in range from 0 to {num_clusters} with step 1
        for idx, value in enumerate(torch.unique(self.cluster_vector)):
            self.cluster_vector = torch.where(self.cluster_vector == value, idx, self.cluster_vector)

        return self.cluster_vector

