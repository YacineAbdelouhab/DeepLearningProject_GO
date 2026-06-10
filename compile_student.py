import tensorflow as tf
import tensorflow.keras as keras


from architecture_student import get_student_model


fixed_lr = 5e-5

optimizer = keras.optimizers.Adam(learning_rate=fixed_lr)

LOGITS = False


# Probits for inference
def get_inference_model(weights_path):

    inference_model = get_student_model(logits=False)
    inference_model.load_weights(weights_path, by_name=True, skip_mismatch=True)
    inference_model.trainable = False
    
    return inference_model


if __name__ == "__main__":
    # --- UTILISATION ---
    # with probits 
    if not LOGITS:
        model = get_inference_model('MODEL_NAME.h5')
        model.compile(
            optimizer=optimizer,
            loss={'policy': 'categorical_crossentropy', 'value': 'mse'},
            loss_weights={'policy': 1.0, 'value': 1.0},
            metrics={'policy': keras.metrics.CategoricalAccuracy(name='categorical_accuracy'), 'value': 'mae'}
        )
        model.summary()
        
    else:
        # with logits
        model = get_student_model()
        model.load_weights('MODEL_NAME.h5') 
        model.trainable = False
        model.compile(
            optimizer=optimizer,
            loss={'policy': tf.keras.losses.CategoricalCrossentropy(from_logits=True), 'value': 'mse'},
            loss_weights={'policy': 1.0, 'value': 1.0},
            metrics={'policy': keras.metrics.CategoricalAccuracy(name='categorical_accuracy'), 'value': 'mae'}
        )

        model.summary()
    