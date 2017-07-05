import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
import pickle
import os
import csv
import utils

# Pass directory of your model.
tf.flags.DEFINE_string("log_dir", './runs/1493109667', "Checkpoint directory")
# Pass name of meta-graph-file in <log_dir>
tf.flags.DEFINE_string("meta_graph_file", 'model-60000.meta', "Name of meta graph file")
tf.flags.DEFINE_integer("batch_size", 32, "Batch size (default: 32)")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nCommand-line Arguments:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

model_id = FLAGS.log_dir.split('/')[-1]
if model_id == '':
    model_id = FLAGS.log_dir.split('/')[-2]

def main(unused_argv):
    # Load test data.
    data = pickle.load(open('./data/a2_dataTest.pkl', 'rb'))
    test_data = utils.crop_mask_images(data, 'rgb', apply_mask=True, mask='segmentation').reshape(-1, utils.IMAGE_HEIGHT, utils.IMAGE_WIDTH, utils.NUM_IMAGE_CLASSES)

    # MNIST data is provided as an example:
    '''
    mnist = learn.datasets.load_dataset("mnist")
    test_data = mnist.test.images.reshape(-1,28,28,1)  # Returns np.array
    test_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    '''

    with tf.Session() as sess:
        # Restore computation graph.
        saver = tf.train.import_meta_graph(os.path.join(FLAGS.log_dir, FLAGS.meta_graph_file))
        # Restore variables.
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.log_dir))
        # Restore ops.
        predictions = tf.get_collection('predictions')[0]
        input_samples_op = tf.get_collection('input_samples_op')[0]
        mode = tf.get_collection('mode')[0]

        def do_prediction(sess, samples):
            batches = utils.data_iterator_samples(samples, FLAGS.batch_size)
            test_predictions = []
            for batch_samples in batches:
                feed_dict = {input_samples_op: batch_samples,
                             mode: False}
                test_predictions.extend(sess.run(predictions, feed_dict=feed_dict))
            return test_predictions

        test_predictions = do_prediction(sess, test_data)
        utils.export_csv(test_predictions)

if __name__ == '__main__':
    tf.app.run()
