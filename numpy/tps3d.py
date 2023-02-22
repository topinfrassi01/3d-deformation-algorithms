#coding=utf-8
import numpy as np

def compute_distance_matrix(points, clip_zero=0.0):
    """
    Computes a symetric matrix containing the distance between each points.
    
    Parameters
    ----------
    points : nd.array, shape [N,3]
        Points to find the matrix for.
     
    Returns
    ----------     
    np.ndarray, shape [N,N]
            Distance matrix between each points
    """
    difference = (np.expand_dims(points, axis=0) - np.expand_dims(points, axis=1))
    return np.clip(np.sqrt(np.einsum('...k,...k->...',difference,difference)), a_min=clip_zero, a_max=None)

# Translated to Python from : https://www.mathworks.com/matlabcentral/fileexchange/37576-3d-thin-plate-spline-warping-function
def tps_3d(points, anchors, anchors_target, anchors_weights=None):
    """
        Uses Thin-Plate Spline interpolation to deform a point cloud.

        Parameters
        ----------
        points : np.ndarray, shape [N,3]
            The point cloud to deform.
        anchors : np.ndarray, shape [K, 3]
            The anchors points are the points that guide the deformation.
        destination : np.ndarray, shape [K,3]
            These are the coordinates of the anchors after deformation.
        Returns
        -------
        np.ndarray, shape [N,3]
            Deformed point cloud according to the anchors movement.
    """
    if anchors_weights is None:
        anchors_weights = compute_distance_matrix(anchors)

    P = np.hstack((np.ones((n_anchors, 1)), anchors))
    L = np.vstack( (np.hstack((anchor_distances, P)), np.hstack((np.transpose(P), np.zeros((4,4))))))

    param = np.matmul(np.linalg.pinv(L), np.vstack((destination, np.zeros((4,3)))))

    n_points = points.shape[0]
    anchor_to_points_distances = np.zeros((n_points, n_anchors))

    gx=points[:,0]
    gy=points[:,1]
    gz=points[:,2]

    for anchor_id in range(n_anchors):
        anchor_to_points_distances[:,anchor_id] = (gx - anchors[anchor_id,0])**2 + (gy - anchors[anchor_id,1])**2 + (gz - anchors[anchor_id,2])**2

    anchor_to_points_distances = np.clip(anchor_to_points_distances, a_min=1e-320, a_max=None)
    anchor_to_points_distances = np.sqrt(anchor_to_points_distances)

    P = np.hstack((np.ones((n_points, 1)), np.expand_dims(gx,1), np.expand_dims(gy,1), np.expand_dims(gz,1)))
    L = np.hstack((anchor_to_points_distances, P))

    return np.matmul(L, param)
