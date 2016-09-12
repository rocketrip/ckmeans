from collections import namedtuple

KmeansResult = namedtuple(
    'ClusterResult',
    ['clustering', 'k', 'centers', 'sizes', 'within_ss', 'total_ss', 'between_ss']
)

KmeansResult.__doc__ = """Cluster result container

Attributes
----------
clustering : np.array
    A 1-D array the same length as x, indicating cluster membership (starting at 0)
centers : np.array
    A 1-D array containing cluster centers
sizes : np.array
    A 1-D array containing the number of points in each cluster
within_ss : np.array
    A 1-D array containing within-cluster sums of squares
total_ss : number
    The total sum of squares
between_ss : number
    The between-cluster sum of squares
"""
