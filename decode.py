from __future__ import print_function
import io
import sys
import numpy as np
from keras.preprocessing.image import save_img
import tensorflow as tf
import tensorflow.compat.v1 as tf
from model import decoder

def decode(inp1,filename1):
    tf.compat.v1.disable_eager_execution()
    flags = tf.app.flags
    flags.DEFINE_integer('iters', 16, 'number of iterations')
    flags.DEFINE_string('output', filename1 + '(compressed).png', 'output path')
    flags.DEFINE_string('model', 'save/model', 'saved model')

    FLAGS = flags.FLAGS
    loaded = np.load(inp1)
        
    
    shape, original_shape = loaded['s'], loaded['o']
    new_height = original_shape[0] + 16 - original_shape[0] % 16
    new_width = original_shape[1] + 16 - original_shape[1] % 16
    codes = np.unpackbits(loaded['c'])
    batch_size = 1
    
    codes = codes.reshape(shape).astype(np.float32) * 2 - 1
    iters = codes.shape[0]
    
    pcodes = tf.placeholder(tf.float32, shape)
    d = decoder(batch_size=batch_size, height=new_height, width=new_width)
    output = tf.constant(0.5)
    
    for i in range(iters):
        output += d.decode(pcodes[i])
        
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, FLAGS.model)
        eval_output = output.eval(feed_dict={pcodes: codes}).clip(0, 1) * 255
        
    output = np.squeeze(eval_output)
    save_img(FLAGS.output, output[:original_shape[0], :original_shape[1]])
    
    
