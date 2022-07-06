# videoclusters

Python library to perform clustering computations on GPU.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install videoclusters.

```bash
pip install videoclusters
```

## Usage example

```python
from videoclusters.centroid_based.KMeans import KMeans

# initialization of the cluster
kmeans_cluster = KMeans(num_clusters=2)

# initialization of uneven distributed data
data = torch.randint(low=1,
                     high=10,
                     size=(100, 2),
                     dtype=torch.float32).to('cuda')

data[50:] = torch.randint(low=20,
                          high=30,
                          size=(50, 2),
                          dtype=torch.float32).to('cuda')

# starting computation
result = kmeans_cluster.fit_predict(data)
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
