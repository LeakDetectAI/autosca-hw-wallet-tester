import tensorflow as tf
from keras import metrics


class MeanRank(metrics.Metric):
    def __init__(self, name='mean_rank', **kwargs):
        super().__init__(name=name, dtype='float32', **kwargs)
        self.ranks = self.add_weight(name='ranks', initializer='zeros', dtype='float32')
        self.count = self.add_weight(name='count', initializer='zeros', dtype='float32')

    def update_state(self, y_true, y_pred, sample_weight=None):
        predictions = tf.argsort(y_pred, direction='DESCENDING', axis=-1)
        if len(y_true.shape) > 1 and y_true.shape[-1] > 1:
            y_true = tf.argmax(y_true, axis=-1)
        y_true = tf.cast(y_true, tf.int32)
        y_true_expanded = tf.expand_dims(y_true, -1)
        is_match = tf.equal(predictions, y_true_expanded)
        indices = tf.where(is_match)
        ranks = tf.cast(indices[:, 1], tf.float32) + 1
        self.ranks.assign_add(tf.reduce_sum(ranks))
        self.count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self):
        return self.ranks / self.count

    def reset_state(self):
        self.ranks.assign(0.)
        self.count.assign(0.)
