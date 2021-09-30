from __future__ import print_function

import io
import numpy as np
import imageio
import tensorflow as tf
import tensorflow.compat.v1 as tf
from model import encoder, decoder


def encode(inp,img):
    tf.compat.v1.disable_eager_execution()
    flags = tf.app.flags
    input_img=imageio.imread(inp)
    flags.DEFINE_integer('iters',16, 'number of iterations')
    flags.DEFINE_string('output', img + '(compressed).npz', 'output')
    flags.DEFINE_string('model', 'save/model', 'saved model')

    FLAGS = flags.FLAGS
    img = input_img.astype(np.float32)
      
    height, width, channel = img.shape
    new_height = height + 16 - height % 16
    new_width = width + 16 - width % 16
    img_padded = np.pad(img, ((0, new_height - height),(0, new_width - width), (0, 0)), 'constant')
    inputs = np.expand_dims(img_padded, axis=0)
    batch_size = 1

    pinputs = tf.placeholder(tf.float32, [batch_size, new_height, new_width, 3])
    inputs_ = pinputs / 255.0 - 0.5

    e = encoder(batch_size=batch_size, height=new_height, width=new_width)
    d = decoder(batch_size=batch_size, height=new_height, width=new_width)

    codes = []

    for i in range(FLAGS.iters):
        code = e.encode(inputs_)
        codes.append(code)
        outputs = d.decode(code)
        inputs_ = inputs_ - outputs
        

    saver = tf.train.Saver()
    eval_codes = []

    with tf.Session() as sess:
        saver.restore(sess, FLAGS.model)
        for i in range(FLAGS.iters):
            c = codes[i].eval(feed_dict={pinputs: inputs})
            eval_codes.append(c)

    int_codes = (np.stack(eval_codes).astype(np.int8) + 1) // 2
    export = np.packbits(int_codes.reshape(-1))

    np.savez_compressed(FLAGS.output, s=int_codes.shape, o=(height, width), c=export)
    
