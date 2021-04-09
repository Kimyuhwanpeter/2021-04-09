# -*- coding:utf-8 -*-
import tensorflow as tf

l2 = tf.keras.regularizers.l2

def GEI_age_model_V7(input_shape=(128, 88, 1), num_classes=86, weight_decay=0.000001):

    h = inputs = tf.keras.Input(input_shape)

    crop_1 = tf.image.crop_to_bounding_box(h, 0, 0, 22, 88)
    crop_1 = tf.image.resize(crop_1, [128, 88])
    crop_2 = tf.image.crop_to_bounding_box(h, 22, 0, 48, 88)
    crop_2 = tf.image.resize(crop_2, [128, 88])
    crop_3 = tf.image.crop_to_bounding_box(h, 70, 0, 58, 88)
    crop_3 = tf.image.resize(crop_3, [128, 88])
    ########################################################################################
    h = tf.keras.layers.ZeroPadding2D((3,3))(h)
    h = tf.keras.layers.Conv2D(filters=32,
                               kernel_size=7,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)

    h = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2, padding="same")(h)

    h = tf.keras.layers.ZeroPadding2D((3,3))(h)
    h = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=7,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)

    h = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2, padding="same")(h)
    ########################################################################################

    ########################################################################################
    crop_1 = tf.keras.layers.ZeroPadding2D((2, 3))(crop_1)
    crop_1 = tf.keras.layers.Conv2D(filters=32,
                                    kernel_size=(5,7),
                                    strides=1,
                                    padding="valid",
                                    use_bias=False,
                                    kernel_regularizer=l2(weight_decay))(crop_1)
    crop_1 = tf.keras.layers.BatchNormalization()(crop_1)
    crop_1 = tf.keras.layers.LeakyReLU()(crop_1)

    crop_1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2, padding="same")(crop_1)

    crop_1 = tf.keras.layers.ZeroPadding2D((2, 3))(crop_1)
    crop_1 = tf.keras.layers.Conv2D(filters=64,
                                    kernel_size=(5,7),
                                    strides=1,
                                    padding="valid",
                                    use_bias=False,
                                    kernel_regularizer=l2(weight_decay))(crop_1)
    crop_1 = tf.keras.layers.BatchNormalization()(crop_1)
    crop_1 = tf.keras.layers.LeakyReLU()(crop_1)

    crop_1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2, padding="same")(crop_1)
    ########################################################################################

    ########################################################################################
    crop_2 = tf.keras.layers.ZeroPadding2D((2, 3))(crop_2)
    crop_2 = tf.keras.layers.Conv2D(filters=32,
                                    kernel_size=(5,7),
                                    strides=1,
                                    padding="valid",
                                    use_bias=False,
                                    kernel_regularizer=l2(weight_decay))(crop_2)
    crop_2 = tf.keras.layers.BatchNormalization()(crop_2)
    crop_2 = tf.keras.layers.LeakyReLU()(crop_2)

    crop_2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2, padding="same")(crop_2)

    crop_2 = tf.keras.layers.ZeroPadding2D((2, 3))(crop_2)
    crop_2 = tf.keras.layers.Conv2D(filters=64,
                                    kernel_size=(5,7),
                                    strides=1,
                                    padding="valid",
                                    use_bias=False,
                                    kernel_regularizer=l2(weight_decay))(crop_2)
    crop_2 = tf.keras.layers.BatchNormalization()(crop_2)
    crop_2 = tf.keras.layers.LeakyReLU()(crop_2)

    crop_2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2, padding="same")(crop_2)
    ########################################################################################

    ########################################################################################
    crop_3 = tf.keras.layers.ZeroPadding2D((2, 3))(crop_3)
    crop_3 = tf.keras.layers.Conv2D(filters=32,
                                    kernel_size=(5,7),
                                    strides=1,
                                    padding="valid",
                                    use_bias=False,
                                    kernel_regularizer=l2(weight_decay))(crop_3)
    crop_3 = tf.keras.layers.BatchNormalization()(crop_3)
    crop_3 = tf.keras.layers.LeakyReLU()(crop_3)

    crop_3 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2, padding="same")(crop_3)

    crop_3 = tf.keras.layers.ZeroPadding2D((2, 3))(crop_3)
    crop_3 = tf.keras.layers.Conv2D(filters=64,
                                    kernel_size=(5,7),
                                    strides=1,
                                    padding="valid",
                                    use_bias=False,
                                    kernel_regularizer=l2(weight_decay))(crop_3)
    crop_3 = tf.keras.layers.BatchNormalization()(crop_3)
    crop_3 = tf.keras.layers.LeakyReLU()(crop_3)

    crop_3 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2, padding="same")(crop_3)
    ########################################################################################

    #crop = tf.concat([crop_1, crop_2, crop_3], -1)
    crop = crop_1 + crop_2 + crop_3
    h = tf.concat([h, crop], -1)

    h = tf.keras.layers.ZeroPadding2D((2,2))(h)
    h = tf.keras.layers.Conv2D(filters=128,
                               kernel_size=5,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)

    h = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2, padding="same")(h)

    h = tf.keras.layers.ZeroPadding2D((2,2))(h)
    h = tf.keras.layers.Conv2D(filters=128,
                               kernel_size=5,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)

    #h = tf.expand_dims(h, 1)    # [None, 1, h, w, c]
    #h = tf.keras.layers.ConvLSTM2D(filters=128,
    #                               kernel_size=(3,3),
    #                               strides=1,
    #                               padding="valid",
    #                               return_sequences=True,
    #                               kernel_regularizer=l2(weight_decay))(h)
    #h = tf.squeeze(h, 1)
    #h = tf.keras.layers.BatchNormalization()(h)
    #h = tf.keras.layers.LeakyReLU()(h)

    h = tf.keras.layers.GlobalMaxPool2D()(h)

    h = tf.keras.layers.Dense(1024)(h)
    h = tf.keras.layers.Dropout(0.5)(h)

    h = tf.keras.layers.Dense(1024)(h)
    h = tf.keras.layers.Dropout(0.5)(h)

    h1 = tf.keras.layers.Dense(num_classes)(h)

    h = tf.keras.layers.Dense(90)(h)

    h = tf.keras.layers.Reshape((9, 10))(h)


    return tf.keras.Model(inputs=inputs, outputs=[h, h1])

class LinearDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    # if `step` < `step_decay`: use fixed learning rate
    # else: linearly decay the learning rate to zero

    def __init__(self, initial_learning_rate, total_steps, step_decay):
        super(LinearDecay, self).__init__()
        self._initial_learning_rate = initial_learning_rate
        self._steps = total_steps
        self._step_decay = step_decay
        self.current_learning_rate = tf.Variable(initial_value=initial_learning_rate, trainable=False, dtype=tf.float32)

    def __call__(self, step):
        self.current_learning_rate.assign(tf.cond(
            step % self._step_decay == 0 and step != 0,
            true_fn=lambda: self.current_learning_rate * 0.1,
            false_fn=lambda: self.current_learning_rate
        ))
        return self.current_learning_rate