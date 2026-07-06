import tensorflow as tf
from keras import backend as K


def binary_crossentropy_focal_loss(alpha=0.25, gamma=2.0, from_logits=False):
    """Focal loss for binary classification."""
    def loss(y_true, y_pred):
        epsilon = K.epsilon()
        if from_logits:
            y_pred = tf.sigmoid(y_pred)
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        modulating_factor = K.pow(1.0 - p_t, gamma)
        return K.mean(alpha_factor * modulating_factor * cross_entropy)
    return loss


def binary_crossentropy_focal_loss_ratio(alpha=0.25, gamma=2.0, from_logits=False, n=1):
    """Focal loss ratio for binary classification."""
    def loss(y_true, y_pred):
        fl = binary_crossentropy_focal_loss(alpha=alpha, gamma=gamma, from_logits=from_logits)
        original = fl(y_true, y_pred)
        shuffled = 0.0
        for _ in range(n):
            shuffled += fl(tf.random.shuffle(y_true), y_pred)
        return original / (shuffled / n)
    return loss


def bce_dice_loss(alpha=0.5, beta=0.5):
    """Combined BCE + Dice loss."""
    def loss(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        bce = K.binary_crossentropy(y_true, y_pred)
        intersection = K.sum(y_true * y_pred, axis=-1)
        dice = 1. - (2. * intersection + 1.) / (K.sum(y_true, axis=-1) + K.sum(y_pred, axis=-1) + 1.)
        return alpha * K.mean(bce) + beta * K.mean(dice)
    return loss
