import tensorflow as tf

class RankBasedStochasticPoolingLayer(tf.keras.layers.Layer):
    def __init__(self, alpha, pool_size=2, stride=2, **kwargs):
        self.alpha = alpha
        self.pool_size = pool_size
        self.stride = stride
        self.pool_area_size = pool_size*pool_size
        self.init_probabilities = [[(alpha*pow(1-alpha, r)) for r in range(self.pool_area_size)]]
        super(RankBasedStochasticPoolingLayer, self).__init__(**kwargs)

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
        self.edge_probs = [[(self.alpha*pow(1-self.alpha, r)) for r in range(self.edge_pool_area_size)]]
        self.br_probs = [[(self.alpha*pow(1-self.alpha, r)) for r in range(self.br_pool_area_size)]]

    def call(self, inputs):
        #self.batch_probs = tf.repeat(self.init_probabilities, repeats=tf.shape(inputs)[0], axis=0)
        @tf.function
        def pool_areas(pool_window):
            """
            Fetch the pool area from the feature map
            Flatten the pool area and sort it in descending order
            Use the multinomial distribution to select, with the probabilities computed beforehand, the output of the pool_area
            Produce a one hot encoding from these result and multiply it with the pool area
            Afterwards sum up the results of the channels, which will be only zeros except the chosen activation
            """
            """
            pool_area = inputs[:,tf.cast(pool_window[0], dtype=tf.int32):tf.cast(pool_window[2], dtype=tf.int32), tf.cast(pool_window[1], dtype=tf.int32):tf.cast(pool_window[3], dtype=tf.int32),:]
            reshaped_area = tf.reshape(pool_area, shape=(-1, self.pool_area_size, pool_area.shape[3]))
            sorted_area = tf.sort(reshaped_area, axis=1, direction="DESCENDING")
            multinomial = tf.random.categorical(tf.math.log(self.batch_probs), pool_area.shape[3])
            one_hot = tf.one_hot(multinomial, self.pool_area_size)
            one_hot = tf.transpose(one_hot, perm=[0,2,1])
            result = tf.multiply(one_hot, sorted_area)
            return tf.reduce_sum(result, axis=1)
            """
            pool_area = inputs[:,tf.cast(pool_window[0], dtype=tf.int32):tf.cast(pool_window[2], dtype=tf.int32), tf.cast(pool_window[1], dtype=tf.int32):tf.cast(pool_window[3], dtype=tf.int32),:]
            pool_area_shape = tf.shape(pool_area, out_type=tf.int32)
            pool_area_size = tf.size(pool_area[0,:,:,0])
            if pool_area_size == self.pool_area_size:
                batch_probs = tf.repeat(self.init_probabilities, repeats=pool_area_shape[0], axis=0)
                flatten_area = tf.reshape(pool_area, shape=(-1, self.pool_area_size, pool_area.shape[3]))
                multinomial = tf.random.categorical(tf.math.log(batch_probs), pool_area_shape[3])
                one_hot = tf.one_hot(multinomial, self.pool_area_size)
                one_hot = tf.transpose(one_hot, perm=[0,2,1])
            elif pool_area_size == self.edge_pool_area_size:
                batch_probs = tf.repeat(self.edge_probs, repeats=pool_area_shape[0], axis=0)
                flatten_area = tf.reshape(pool_area, shape=(-1, self.edge_pool_area_size, pool_area.shape[3]))
                multinomial = tf.random.categorical(tf.math.log(batch_probs), pool_area_shape[3])
                one_hot = tf.one_hot(multinomial, self.edge_pool_area_size)
                one_hot = tf.transpose(one_hot, perm=[0,2,1])
            else:
                batch_probs = tf.repeat(self.br_probs, repeats=pool_area_shape[0], axis=0)
                flatten_area = tf.reshape(pool_area, shape=(-1, self.br_pool_area_size, pool_area.shape[3]))
                multinomial = tf.random.categorical(tf.math.log(batch_probs), pool_area_shape[3])
                one_hot = tf.one_hot(multinomial, self.br_pool_area_size)
                one_hot = tf.transpose(one_hot, perm=[0,2,1])
            sorted_area = tf.sort(flatten_area, axis=1, direction="DESCENDING")
            result = tf.multiply(one_hot, sorted_area)
            return(tf.reduce_sum(result, axis=1))

        output_shape = [-1, tf.cast(inputs.shape[1]/self.stride, dtype=tf.int32), tf.cast(inputs.shape[2]/self.stride, dtype=tf.int32), inputs.shape[3]]
        output = tf.map_fn(pool_areas, self.tf_areas, fn_output_signature=tf.float32)
        output = tf.transpose(output, perm=[1,0,2])
        output = tf.reshape(output, shape=output_shape)
        return output

    @tf.function
    def rbs_pooling(self, feature_map, areas):
        def pool_areas(pool_window):
            """
            Fetch the pool area from the feature map
            Flatten the pool area and sort it in descending order
            Use the multinomial distribution to select, with the probabilities computed beforehand, the output of the pool_area
            Produce a one hot encoding from these result and multiply it with the pool area
            Afterwards sum up the results of the channels, which will be only zeros except the chosen activation
            """
            pool_area = feature_map[pool_window[0]:pool_window[2], pool_window[1]:pool_window[3], :]
            reshaped_area = tf.reshape(pool_area, shape=(self.pool_area_size, pool_area.shape[2]))
            sorted_area = tf.sort(reshaped_area, axis=0, direction="DESCENDING")
            multinomial = tf.random.categorical(tf.math.log(self.probabilities), pool_area.shape[2])
            multinomial_reshaped = tf.reshape(multinomial, shape=[pool_area.shape[2]])
            one_hot = tf.one_hot(multinomial_reshaped, self.pool_area_size)
            one_hot = tf.transpose(one_hot)
            result = tf.multiply(one_hot, sorted_area)
            return tf.reduce_sum(result, axis=0)

        pooled_features = tf.stack([[pool_areas(x) for x in row] for row in areas])
        return pooled_features
            
            

    def compute_output_shape(self, input_shape):
        output_shape = self.call(tf.zeros(input_shape))
        return output_shape.shape

    def get_config(self):
        config = super().get_config()
        config.update({
            "alpha": self.alpha,
            "pool_size": self.pool_size,
            "stride": self.stride
        })
        return config