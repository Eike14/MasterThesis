import tensorflow as tf
import numpy as np

class DropoutMaxPoolingLayer(tf.keras.layers.Layer):
    def __init__(self, drop_rate, pool_size=2, stride=2, **kwargs):
        self.drop_rate = drop_rate
        self.retain_rate = 1-drop_rate
        self.pool_size=pool_size
        self.stride=stride
        self.pool_area_size = pool_size*pool_size
        self.init_probabilities = [[[(self.retain_rate * pow(self.drop_rate, self.pool_area_size-1-k))] for k in range(self.pool_area_size)]]
        super(DropoutMaxPoolingLayer, self).__init__(**kwargs)

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
        self.channel_probs = tf.repeat(self.init_probabilities, repeats=input_shape[3], axis=2)
        self.edge_pool_window = self.tf_areas[int(input_shape[1]/self.stride)-1]
        self.edge_pool_area_size = int((self.edge_pool_window[3] - self.edge_pool_window[1]) * self.pool_size)
        self.br_pool_window = self.tf_areas[-1]
        self.br_pool_area_size = int((self.br_pool_window[3]- self.br_pool_window[1]) * (self.br_pool_window[2] - self.br_pool_window[0]))
        self.edge_probs = [[[(self.retain_rate * pow(self.drop_rate, self.edge_pool_area_size-1-k))] for k in range(self.edge_pool_area_size)]]
        self.br_probs = [[[(self.retain_rate * pow(self.drop_rate, self.br_pool_area_size-1-k))] for k in range(self.br_pool_area_size)]]
        #self.final_probs = tf.repeat(channel_probs, repeats=tf.shape(input_shape)[0], axis=0)

    
    def call(self, inputs, training=None):
        """
        number_of_steps_height = int(inputs.shape[1]/self.pool_size)
        number_of_steps_width = int(inputs.shape[2]/self.pool_size)
        
        #Array for coordinates of all pooling windows
        areas = [[(
            h * self.pool_size,
            w * self.pool_size,
            (h+1) * self.pool_size,
            (w+1) * self.pool_size,
        )
         for w in range(number_of_steps_width)]
        for h in range(number_of_steps_height)]

        tf_areas = tf.convert_to_tensor(np.array(areas), dtype=tf.float32)
        """

        """
        if (Training):
            max pooling dropout
        else:
            probabilistic weighted pooling
        """
        @tf.function
        def pool_areas_dropoutMaxPooling(pool_window):
            #fetches the poolarea from the feature map            
            pool_area = inputs[:,tf.cast(pool_window[0], dtype=tf.int32):tf.cast(pool_window[2], dtype=tf.int32), tf.cast(pool_window[1], dtype=tf.int32):tf.cast(pool_window[3], dtype=tf.int32),:]
            #Performs Dropout on the pool area
            dropout_area = tf.nn.dropout(pool_area, rate=0.5)
            #Takes the maximum of the remaining values and rescales it to its original value
            dropout_max = tf.reduce_max(dropout_area, axis=[1,2])*self.retain_rate
            return dropout_max


        @tf.function
        def pool_areas_PWP(pool_window):
            pool_area = inputs[:,tf.cast(pool_window[0], dtype=tf.int32):tf.cast(pool_window[2], dtype=tf.int32), tf.cast(pool_window[1], dtype=tf.int32):tf.cast(pool_window[3], dtype=tf.int32),:]
            pool_area_shape = tf.shape(pool_area, out_type=tf.int32)
            pool_area_size = tf.size(pool_area[0,:,:,0])
            if pool_area_size == self.pool_area_size:
                channel_probs = tf.repeat(self.init_probabilities, repeats=pool_area.shape[3], axis=2)
                flatten_area = tf.reshape(pool_area, shape=(-1, self.pool_area_size, pool_area.shape[3]))
            elif pool_area_size == self.edge_pool_area_size:
                channel_probs = tf.repeat(self.edge_probs, repeats=pool_area.shape[3], axis=2)
                flatten_area = tf.reshape(pool_area, shape=(-1, self.edge_pool_area_size, pool_area.shape[3]))
            else:
                channel_probs = tf.repeat(self.br_probs, repeats=pool_area.shape[3], axis=2)
                flatten_area = tf.reshape(pool_area, shape=(-1, self.br_pool_area_size, pool_area.shape[3]))
            sorted_area = tf.sort(flatten_area, axis=1)
            final_probs = tf.repeat(channel_probs, repeats=pool_area_shape[0], axis=0)
            scaled_area = tf.multiply(sorted_area, final_probs)
            return(tf.reduce_sum(scaled_area, axis=1))
        
    
        #output = tf.map_fn(max_pooling_dropout_batch, inputs, fn_output_signature=tf.float32, parallel_iterations=50)
        output_shape = [-1, tf.cast(inputs.shape[1]/self.stride, dtype=tf.int32), tf.cast(inputs.shape[2]/self.stride, dtype=tf.int32), inputs.shape[3]]
        if(training):
            output = tf.map_fn(pool_areas_dropoutMaxPooling, self.tf_areas, fn_output_signature=tf.float32)
        else:
            output = tf.map_fn(pool_areas_PWP, self.tf_areas, fn_output_signature=tf.float32)
            
        output = tf.transpose(output, perm=[1,0,2])
        output = tf.reshape(output, shape=output_shape)
        return output
    

    @tf.function
    def max_pooling_dropout(self, feature_map):
        """
        Function for pooling one pooling window in training
        Performs Dropout on the pooling window and takes the largest remaining value
        Tensorflows Dropout function scales the remaining values with 1/retain_rate
        This is not wanted for this function, thats why we rescale the value
        """
        
        def pool_areas(pool_window):
            #fetches the poolarea from the feature map            
            pool_area = feature_map[tf.cast(pool_window[0], dtype=tf.int32):tf.cast(pool_window[2], dtype=tf.int32), tf.cast(pool_window[1], dtype=tf.int32):tf.cast(pool_window[3], dtype=tf.int32),:]
            #Performs Dropout on the pool area
            dropout_area = tf.nn.dropout(pool_area, rate=0.5)
            #Takes the maximum of the remaining values and rescales it to its original value
            dropout_max = tf.reduce_max(dropout_area, axis=[0,1])*self.retain_rate
            return dropout_max
        
        
        pooled_features = tf.stack(tf.map_fn(pool_areas, self.tf_areas, parallel_iterations=50), axis=1)
        #pooled_features = tf.reshape(pooled_features, shape=[int(feature_map.shape[0]/self.pool_size), int(feature_map.shape[1]/self.pool_size), feature_map.shape[2]])
        
        return pooled_features
        #pooled_features = tf.zeros([0, tf.cast(feature_map.shape[1]/self.pool_size, dtype=tf.int32),feature_map.shape[2]])
        #Iterate over all pooling windows
        #for row in tf.range(len(self.areas)):
            #tf.autograph.experimental.set_loop_options(shape_invariants=[pooled_features, tf.TensorShape([None, pooled_features.shape[1], pooled_features.shape[2]])])
            #pooled_features = tf.concat([pooled_features, tf.expand_dims(pool_row(self.areas[row]), axis=0)], axis=0)
        #pooled_features = tf.stack([[pool_areas(x) for x in row] for row in areas])

        """
        i = tf.constant(0)
        while (i < len(self.areas)):
            tf.autograph.experimental.set_loop_options(shape_invariants=[pooled_features, tf.TensorShape([None, pooled_features.shape[1], pooled_features.shape[2]])])
            pooled_features = tf.concat([pooled_features, tf.expand_dims(pool_row(self.tf_areas[i]), axis=0)], axis=0)
            i += 1
        """
        
    
    
    def probabilistic_weighted_pooling(self, feature_map):
        """
        Function for pooling one pooling window in testing
        Produces the pooling output by summing up the all pooling values multiplied by their probabilities
        of being chosen as the output during training time
        """
        
        def pool_areas(pool_window):
            pool_area = feature_map[tf.cast(pool_window[0], dtype=tf.int32):tf.cast(pool_window[2], dtype=tf.int32), tf.cast(pool_window[1], dtype=tf.int32):tf.cast(pool_window[3], dtype=tf.int32),:]
            flatten_area = tf.reshape(pool_area, shape=(self.pool_area_size, pool_area.shape[2]))
            sorted_area = tf.sort(flatten_area, axis=0)
            repeated_probs = tf.repeat(self.probabilities, repeats=pool_area.shape[2], axis=1)
            scaled_area = tf.multiply(sorted_area, repeated_probs)
            return(tf.reduce_sum(scaled_area, axis=0))
        
        def pool_row(row):
            return tf.map_fn(pool_areas, row, parallel_iterations=100)
        
        pooled_features = tf.stack(tf.map_fn(pool_row, self.tf_areas), axis=1)
        #pooled_features = tf.stack([[pool_areas(x) for x in row] for row in areas])
        return pooled_features
            

    def compute_output_shape(self, input_shape):
        output_shape = self.call(tf.zeros(input_shape))
        return output_shape.shape
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "drop_rate": self.drop_rate,
            "pool_size": self.pool_size,
            "stride": self.stride
        })
        return config