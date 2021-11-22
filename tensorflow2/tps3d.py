import tensorflow as tf

def tps_33(anchors, destination, points):
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