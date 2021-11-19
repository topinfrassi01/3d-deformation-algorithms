#coding=utf-8
import numpy as np

# Translated to Python from : https://www.mathworks.com/matlabcentral/fileexchange/37576-3d-thin-plate-spline-warping-function
def thin_plate_spline(anchors, destination, points):
    """
        Uses Thin-Plate Spline interpolation to deform a point cloud.

        Parameters
        ----------
        anchors : nd.array
            The anchors points are the points that guide the deformation.
        destination : nd.array
            These are the coordinates of the anchors after deformation. 
            * The shape of destination and anchors should be the same and their indices should imply correspondance between points.
        points : nd.array
            The point cloud to deform.
        Returns
        -------
        nd.array
            Deformed point cloud according to the anchors movement.
    """
    n_anchors = anchors.shape[0]
    
    anchor_distances = np.zeros((n_anchors, n_anchors))

    for row in range(n_anchors):
        for col in range(row, n_anchors):
            anchor_distances[row,col] = np.sum((anchors[row,:] - anchors[col,:])**2)
            anchor_distances[col,row] = anchor_distances[row,col]

    anchor_distances = np.clip(anchor_distances, a_min=1e-320, a_max=None)
    anchor_distances = np.sqrt(anchor_distances)

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