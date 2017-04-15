import tensorflow as tf
from models.densenet import ops
class OpsTests(tf.test.TestCase):
    def setUp(self):
        self.batch_size= 10
        self.sequence_length = 256
        self.hidden_size = 8
        self.data = tf.ones(dtype=tf.float32,shape=[self.batch_size,self.sequence_length,self.hidden_size])
    def test_avg1d(self):
        with self.test_session():
            output = ops.avg_pool1d(self.data,2)
            self.assertEqual([self.batch_size,self.sequence_length/2,self.hidden_size],output.get_shape().as_list())

            output = ops.avg_pool1d(self.data, self.sequence_length)
            self.assertEqual([self.batch_size, 1, self.hidden_size],
                             output.get_shape().as_list())
    def test_composite_function(self):
        with self.test_session():
            with tf.variable_scope("case1"):
                output = ops.composite_function(self.data,outSize=self.hidden_size,width=3)
                self.assertEqual([self.batch_size, self.sequence_length , self.hidden_size],
                             output.get_shape().as_list())
        with tf.variable_scope("case2"):
            output = ops.composite_function(self.data,outSize=self.hidden_size,width=5)
            self.assertEqual([self.batch_size, self.sequence_length , self.hidden_size],
                             output.get_shape().as_list())
        with tf.variable_scope("case3"):
            output = ops.composite_function(self.data,outSize=self.hidden_size//2,width=5)
            self.assertEqual([self.batch_size, self.sequence_length , self.hidden_size//2],
                             output.get_shape().as_list())

    def test_conv1d(self):
        with self.test_session():
            output = ops.conv1d(self.data, outSize=self.hidden_size, width=3)
            self.assertEqual([self.batch_size, self.sequence_length, self.hidden_size],
                             output.get_shape().as_list())

    def test_bottleneck(self):
        with self.test_session():
            with tf.variable_scope("bottlneck_case"):
                output = ops.bottleneck(self.data,4)
                self.assertEqual([self.batch_size, self.sequence_length, 16],
                                 output.get_shape().as_list())

    def test_makeblock(self):
        with self.test_session():
            with tf.variable_scope("block_layer"):
                growth_rate = 8
                num_layers=5
                output = ops.makeBlock(self.data,growth_rate,num_layers,False)

                self.assertEqual([self.batch_size, self.sequence_length, self.hidden_size+(growth_rate*num_layers)],
                                 output.get_shape().as_list())

            with tf.variable_scope("block_layer_bottled"):
                output = ops.makeBlock(self.data,growth_rate,num_layers,True)
                self.assertEqual([self.batch_size, self.sequence_length, growth_rate*4],
                                 output.get_shape().as_list())


    def test_transition_layer(self):
        output = ops.transition_layer(self.data,1)
        self.assertEqual([self.batch_size, self.sequence_length//2, self.hidden_size],
                                 output.get_shape().as_list())
        with tf.variable_scope("compression_case"):
            output = ops.transition_layer(self.data,0.5)
            self.assertEqual([self.batch_size, self.sequence_length//2, self.hidden_size//2],
                                     output.get_shape().as_list())


    def test_transition_to_vector(self):
        output = ops.transition_to_vector(self.data)
        self.assertEqual([self.batch_size,self.hidden_size],output.get_shape().as_list())

if __name__ == '__main__':
    tf.test.main()
