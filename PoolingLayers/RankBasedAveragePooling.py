import tensorflow as tf

class RankBasedAveragePoolingLayer(tf.keras.layers.Layer):
    def __init__(self, threshold, pool_size=2, stride=2, **kwargs):
        self.threshold = threshold
        self.pool_size = pool_size
        self.stride = stride
        self.pool_area_size = pool_size*pool_size
        super(RankBasedAveragePoolingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        number_of_steps_height = int(input_shape[1]/self.stride)
        number_of_steps_width = int(input_shape[2]/self.stride)
        self.areas = [(
            h * self.stride,
            w * self.stride,
            h * self.stride + self.pool_size if h+1<number_of_steps_height else input_shape[1],
            w * self.stride + self.pool_size if w+1<number_of_steps_width else input_shape[2],
        )
         for h in range(number_of_steps_width)
        for w in range(number_of_steps_height)]

        self.tf_areas = tf.constant(self.areas, dtype=tf.float32)

        self.edge_pool_window = self.tf_areas[int(input_shape[1]/self.stride)-1]
        self.edge_pool_area_size = int((self.edge_pool_window[3] - self.edge_pool_window[1]) * self.pool_size)
        self.br_pool_window = self.tf_areas[-1]
        self.br_pool_area_size = int((self.br_pool_window[3]- self.br_pool_window[1]) * (self.br_pool_window[2] - self.br_pool_window[0]))

    def call(self, inputs):

        @tf.function
        def pool_areas(pool_window):
            pool_area = inputs[:,tf.cast(pool_window[0], dtype=tf.int32):tf.cast(pool_window[2], dtype=tf.int32), tf.cast(pool_window[1], dtype=tf.int32):tf.cast(pool_window[3], dtype=tf.int32),:]
            pool_area_shape = tf.shape(pool_area, out_type=tf.int32)
            #if pool_area.shape[1] is not None:
                #if pool_area.shape[2] is not None:
                    #self.pool_area_size = pool_area.shape[1] * pool_area.shape[2]
            pool_area_size = tf.size(pool_area[0,:,:,0])
            if pool_area_size == self.pool_area_size:
                #channel_probs = tf.repeat(self.init_probabilities, repeats=pool_area.shape[3], axis=2)
                flatten_area = tf.reshape(pool_area, shape=(-1, self.pool_area_size, pool_area.shape[3]))
            elif pool_area_size == self.edge_pool_area_size:
                #channel_probs = tf.repeat(self.edge_probs, repeats=pool_area.shape[3], axis=2)
                flatten_area = tf.reshape(pool_area, shape=(-1, self.edge_pool_area_size, pool_area.shape[3]))
            else:
                #channel_probs = tf.repeat(self.br_probs, repeats=pool_area.shape[3], axis=2)
                flatten_area = tf.reshape(pool_area, shape=(-1, self.br_pool_area_size, pool_area.shape[3]))
            sorted_area = tf.sort(flatten_area, direction="DESCENDING", axis=1)
            #final_probs = tf.repeat(channel_probs, repeats=pool_area_shape[0], axis=0)
            #scaled_area = tf.multiply(sorted_area, final_probs)
            sliced_area = sorted_area[:,0:self.threshold,:]
            return(tf.reduce_mean(sliced_area, axis=1))

        output_shape = [-1, tf.cast(inputs.shape[1]/self.stride, dtype=tf.int32), tf.cast(inputs.shape[2]/self.stride, dtype=tf.int32), inputs.shape[3]]
        output = tf.map_fn(pool_areas, self.tf_areas, fn_output_signature=tf.float32)
        output = tf.transpose(output, perm=[1,0,2])
        output = tf.reshape(output, shape=output_shape)
        return output


    def rbapooling(self, feature_map, areas):
        """
        Sort values in descending order and rank them with 1 for the highest and n for the lowest value
        Compute the average of all values until rank threshold
        """

        def pool_areas(pool_window):
            pool_area = feature_map[pool_window[0]:pool_window[2], pool_window[1]:pool_window[3],:]
            reshaped_pool_area = tf.reshape(pool_area, shape=(self.pool_area_size, pool_area.shape[2]))
            sorted_pool_area = tf.sort(reshaped_pool_area, axis=0, direction="DESCENDING")
            sliced_pool_area = sorted_pool_area[0:self.threshold, :]
            return tf.reduce_mean(sliced_pool_area, axis=0)

        pooled_features = tf.stack([[pool_areas(x) for x in row] for row in areas])
        return pooled_features
            


    def compute_output_shape(self, input_shape):
        output_shape = self.call(tf.zeros(input_shape))
        return output_shape.shape
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "threshold": self.threshold,
            "pool_size": self.pool_size,
            "stride": self.stride
        })
        return config