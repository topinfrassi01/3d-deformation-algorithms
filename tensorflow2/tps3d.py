import tensorflow as tf

def tps_3d_batch(anchors, destination, points):
    """
        Uses Thin-Plate Spline interpolation to deform a point cloud.

        Parameters
        ----------
        anchors : Tensor [Nx3]
            The anchors points are the points that guide the deformation.
        destination : Tensor [N,3]
            These are the coordinates of the anchors after deformation. 
            * The shape of destination and anchors should be the same and their indices should imply correspondance between points.
        points : Tensor [M,3] with M > N
            The point cloud to deform.
        Returns
        -------
        Tensor
            Deformed point cloud according to the anchors movement.
    """

    anchors = tf.convert_to_tensor(anchors, 'float64')
    destination = tf.convert_to_tensor(destination, 'float64')
    points = tf.convert_to_tensor(points, 'float64')

    n_batches = tf.shape(anchors)[0]
    n_anchors = tf.shape(anchors)[1]
    K = tf.zeros((n_batches, n_anchors, n_anchors), 'float64')

    distance_maps = (tf.expand_dims(anchors, axis=-2) - tf.expand_dims(anchors, axis=-3))
    distance_maps = tf.einsum("...i,...i->...", distance_maps, distance_maps)

    distance_maps = tf.math.maximum(distance_maps, 1e-320)
    distance_maps = tf.sqrt(distance_maps)

    P = tf.concat((tf.ones((n_batches, n_anchors, 1), 'float64'), anchors), axis=-1)
    L = tf.concat((tf.concat((distance_maps, P), axis=-1), tf.concat((tf.transpose(P, perm=[0,2,1]), tf.zeros((n_batches, 4, 4), 'float64')), axis=-1)), axis=-2)

    param = tf.matmul(tf.linalg.pinv(L), tf.concat((destination, tf.zeros((n_batches, 4,3), 'float64')), axis=-2))

    n_points = tf.shape(points)[1]


    K = (tf.expand_dims(points, axis=-2) - tf.expand_dims(anchors, axis=-3))
    K = tf.einsum("...i,...i->...", K, K)

    K = tf.math.maximum(K, 1e-320)
    K = tf.sqrt(K)

    P = tf.concat(( tf.ones((n_batches, n_points, 1), 'float64'), points), axis=-1)
    L = tf.concat((K, P), axis=-1)

    return tf.matmul(L, param)


def tps_3d(anchors, destination, points):
    """
        Uses Thin-Plate Spline interpolation to deform a point cloud.

        Parameters
        ----------
        anchors : Tensor [Nx3]
            The anchors points are the points that guide the deformation.
        destination : Tensor [N,3]
            These are the coordinates of the anchors after deformation. 
            * The shape of destination and anchors should be the same and their indices should imply correspondance between points.
        points : Tensor [M,3] with M > N
            The point cloud to deform.
        Returns
        -------
        Tensor
            Deformed point cloud according to the anchors movement.
    """

    anchors = tf.convert_to_tensor(anchors, 'float64')
    destination = tf.convert_to_tensor(destination, 'float64')
    points = tf.convert_to_tensor(points, 'float64')

    n_anchors = tf.shape(anchors)[0]
    K = tf.zeros((n_anchors, n_anchors), 'float64')

    distance_maps = (tf.expand_dims(anchors, axis=-2) - tf.expand_dims(anchors, axis=-3))
    distance_maps = tf.einsum("...i,...i->...", distance_maps, distance_maps)

    distance_maps = tf.math.maximum(distance_maps, 1e-320)
    distance_maps = tf.sqrt(distance_maps)

    P = tf.concat((tf.ones((n_anchors, 1), 'float64'), anchors), axis=1)
    L = tf.concat((tf.concat((distance_maps, P), axis=1), tf.concat((tf.transpose(P), tf.zeros((4,4), 'float64')), axis=1)), axis=0)

    param = tf.matmul(tf.linalg.pinv(L), tf.concat((destination, tf.zeros((4,3), 'float64')), axis=0))

    n_points = tf.shape(points)[0]


    K = (tf.expand_dims(points, axis=-2) - tf.expand_dims(anchors, axis=-3))
    K = tf.einsum("...i,...i->...", K, K)

    K = tf.math.maximum(K, 1e-320)
    K = tf.sqrt(K)

    P = tf.concat(( tf.ones((n_points, 1), 'float64'), points), axis=1)
    L = tf.concat((K, P), axis=1)

    return tf.matmul(L, param)