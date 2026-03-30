import tensorflow as tf
import tensorflow.keras as keras


from architecture_student import get_student_model


fixed_lr = 5e-5

optimizer = keras.optimizers.Adam(learning_rate=fixed_lr)

model = get_student_model()
model.load_weights('MODEL_NAME.h5') # Nom du fichier h5 sur le drive
model.trainable = False


model.compile(
    optimizer=optimizer,
    loss={'policy': tf.keras.losses.CategoricalCrossentropy(from_logits=True), 'value': 'mse'},
    loss_weights={'policy': 1.0, 'value': 0.5},
    metrics={'policy': keras.metrics.CategoricalAccuracy(name='categorical_accuracy'), 'value': 'mae'}
)

model.summary()