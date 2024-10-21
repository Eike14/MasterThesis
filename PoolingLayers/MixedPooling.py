import tensorflow as tf 

class MixedPoolingLayer(tf.keras.layers.Layer):
    def __init__(self, pool_size=2, stride=2, compute_output_shape_function=None, **kwargs):
        self.compute_output_shape_function = compute_output_shape_function
        self.pool_size = pool_size
        self.stride = stride
        self.pool_area_size = pool_size*pool_size
        super(MixedPoolingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.frequencies_Max = tf.Variable(tf.zeros(shape=[input_shape[3]], dtype=tf.float64), trainable=False)
        self.frequencies_Avg = tf.Variable(tf.zeros(shape=[input_shape[3]], dtype=tf.float64), trainable=False)

    def call(self, inputs, training=None):
        
        #Area for coordinates for all pooling window
        #Tuple: (y_min, x_min, y_max, x_max)
        number_of_steps_height = int(inputs.shape[1]/self.pool_size)
        number_of_steps_width = int(inputs.shape[2]/self.pool_size)
        
        areas = [[(
            h * self.pool_size,
            w * self.pool_size,
            (h+1) * self.pool_size,
            (w+1) * self.pool_size,
        )
         for w in range(number_of_steps_width)]
        for h in range(number_of_steps_height)]

        
        def mixed_pooling_batch(feature_map):
            if(training):
                return self.mixed_pooling_training(feature_map, areas)
            else:
                return self.mixed_pooling_testing(feature_map, areas)
        
        #Calls the pooling operation for the 3D input over the whole batch size
        output = tf.map_fn(mixed_pooling_batch, inputs, fn_output_signature=tf.float32)
        return output
            
    def mixed_pooling_training(self, feature_map, areas):
        #Support Function for pooling on one specific pooling window
        def pool_areas(pool_window):
            pool_area = feature_map[pool_window[0]:pool_window[2], pool_window[1]:pool_window[3],:]
            k = tf.cast(tf.random.uniform(shape=[feature_map.shape[2]], minval=0, maxval=2, dtype=tf.int32), dtype=tf.float32)
            self.frequencies_Max.assign_add(tf.cast(k, dtype=tf.float64))
            j = 1-k
            self.frequencies_Avg.assign_add(tf.cast(j, dtype=tf.float64))
            result = j * tf.reduce_max(pool_area, axis=[0,1]) + k * tf.reduce_mean(pool_area, axis=[0,1])             
            return result
        #Calls the pooling function on every pooling window and stacks it inside a tensor
        pooled_features = tf.stack([[pool_areas(x) for x in row] for row in areas])
        return pooled_features

    def mixed_pooling_testing(self, feature_map, areas):
        freq_comp = tf.transpose(tf.cast(self.frequencies_Max >= self.frequencies_Avg, dtype=tf.float32))
        def pool_areas(pool_window):
            pool_area = feature_map[pool_window[0]:pool_window[2], pool_window[1]:pool_window[3],:]
            result = freq_comp * tf.reduce_max(pool_area, axis=[0,1]) + (1-freq_comp) * tf.reduce_mean(pool_area, axis=[0,1])
            return result

        pooled_features = tf.stack([[pool_areas(x) for x in row] for row in areas])
        return pooled_features
            
        
    def compute_output_shape(self, input_shape):
        if self._compute_output_shape_function:
            return self._compute_output_shape_function(input_shape)
        else:
            # If compute_output_shape_function is not provided, infer output shape from the call function
            output_shape = self.call(tf.zeros(input_shape))
            return output_shape.shape
    
    def get_config(self):
        return super().get_config()
    
    @classmethod
    def from_config(cls, config):
        return super().from_config(config)