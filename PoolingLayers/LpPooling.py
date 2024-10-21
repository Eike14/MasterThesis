import tensorflow as tf

class LpPoolingLayer(tf.keras.layers.Layer):
    def __init__(self, pool_size=2, stride=2, **kwargs):
        self.pool_size=pool_size
        self.stride=stride
        self.pool_area_size=pool_size*pool_size
        super(LpPoolingLayer,self).__init__(**kwargs)

    def build(self, input_shape):
        bias_shape = (input_shape[1], input_shape[2], input_shape[3])
        self.biases = self.add_weight(
            shape=bias_shape,
            trainable=True,
            name="biases"
        )

        self.p = self.add_weight(
            shape=(int(input_shape[1]/self.stride), int(input_shape[2]/self.stride), input_shape[3]),
            trainable=True,
            name="orders"
<<<<<<< HEAD
        )
        
        number_of_steps_height = int(input_shape[1]/self.stride)
        number_of_steps_width = int(input_shape[2]/self.stride)
        self.areas = [(
            h * self.stride,
            w * self.stride,
            h * self.stride + self.pool_size if h+1<number_of_steps_height else input_shape[1],
            w * self.stride + self.pool_size if w+1<number_of_steps_width else input_shape[2],
        )
=======
        )
        
        number_of_steps_height = int(input_shape[1]/self.pool_size)
        number_of_steps_width = int(input_shape[2]/self.pool_size)
        self.areas = [(
            h * self.pool_size,
            w * self.pool_size,
            (h+1) * self.pool_size,
            (w+1) * self.pool_size,
        )
>>>>>>> 0320eb77585aefd3f33e868fa588b6391ccdb9b4
         for h in range(number_of_steps_width)
        for w in range(number_of_steps_height)]

        self.tf_areas = tf.constant(self.areas, dtype=tf.float32)

    def call(self, inputs):
        
<<<<<<< HEAD
=======

        

>>>>>>> 0320eb77585aefd3f33e868fa588b6391ccdb9b4
        def pool_areas(pool_window):
            pool_area = inputs[:,tf.cast(pool_window[0], dtype=tf.int32):tf.cast(pool_window[2], dtype=tf.int32), tf.cast(pool_window[1], dtype=tf.int32):tf.cast(pool_window[3], dtype=tf.int32),:]
            p = self.p[tf.cast((pool_window[0]/self.stride), dtype=tf.int32),tf.cast((pool_window[1]/self.stride), dtype=tf.int32),:]
            p = 1 + tf.math.log(1+tf.math.exp(p))
<<<<<<< HEAD
            pool_area_shape = tf.shape(pool_area, out_type=tf.int32)
            batch_p = tf.expand_dims(p, axis=0)
            batch_p = tf.repeat(batch_p, repeats=tf.shape(pool_area)[0], axis=0)
            p_pow = tf.reshape(batch_p, shape=(-1,1,1,batch_p.shape[1]))
            p_pow = tf.repeat(p_pow, repeats=pool_area_shape[1], axis=1)
            p_pow = tf.repeat(p_pow, repeats=pool_area_shape[2], axis=2)
=======
            batch_p = tf.expand_dims(p, axis=0)
            batch_p = tf.repeat(batch_p, repeats=tf.shape(pool_area)[0], axis=0)
            p_pow = tf.reshape(batch_p, shape=(-1,1,1,batch_p.shape[1]))
            p_pow = tf.repeat(p_pow, repeats=2, axis=1)
            p_pow = tf.repeat(p_pow, repeats=2, axis=2)
>>>>>>> 0320eb77585aefd3f33e868fa588b6391ccdb9b4
            biases = self.biases[int(pool_window[0]):int(pool_window[2]),int(pool_window[1]):int(pool_window[3]),:]
            batch_biases = tf.expand_dims(biases, axis=0)
            batch_biases = tf.repeat(batch_biases, repeats=tf.shape(pool_area)[0], axis=0)
            difference = tf.abs(pool_area - batch_biases)
            exp = tf.math.pow(difference, p_pow)
            mean = tf.reduce_mean(exp, axis=[1,2])
            result = tf.math.pow(mean, 1/p)
            return result


<<<<<<< HEAD
        output_shape = [-1, tf.cast(inputs.shape[1]/self.stride, dtype=tf.int32), tf.cast(inputs.shape[2]/self.stride, dtype=tf.int32), inputs.shape[3]]
=======
        output_shape = [-1, tf.cast(inputs.shape[1]/self.pool_size, dtype=tf.int32), tf.cast(inputs.shape[2]/self.pool_size, dtype=tf.int32), inputs.shape[3]]
>>>>>>> 0320eb77585aefd3f33e868fa588b6391ccdb9b4
        output = tf.map_fn(pool_areas, self.tf_areas, fn_output_signature=tf.float32)
        output = tf.transpose(output, perm=[1,0,2])
        output = tf.reshape(output, shape=output_shape)
        return output

    def lpPooling(self, feature_map, areas):
        def pool_areas(pool_window):
            pool_area = feature_map[pool_window[0]:pool_window[2],pool_window[1]:pool_window[3],:]
            p = self.p[int(pool_window[0]/self.stride),int(pool_window[1]/self.stride),:]
            p = 1 + tf.math.log(1+tf.math.exp(p))
            biases = self.biases[pool_window[0]:pool_window[2],pool_window[1]:pool_window[3],:]
            difference = tf.abs(pool_area - biases)
            exp = tf.math.pow(p, difference)
            mean = tf.reduce_mean(exp, axis=[0,1])
            result = tf.math.pow(mean, 1/p)
            return result

        pooled_features = tf.stack([[pool_areas(x) for x in row] for row in areas])
        return pooled_features
            
    def compute_output_shape(self, input_shape):
        output_shape = self.call(tf.zeros(input_shape))
        return output_shape.shape
<<<<<<< HEAD
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "pool_size": self.pool_size,
            "stride": self.stride
        })
        return config
=======
>>>>>>> 0320eb77585aefd3f33e868fa588b6391ccdb9b4
    