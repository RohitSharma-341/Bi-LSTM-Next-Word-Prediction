import tensorflow as tf

def check_gpu():
    # Check TensorFlow version
    print("TensorFlow Version:", tf.__version__)

    # Check for available GPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if not gpus:
        print("No GPU devices found")
    else:
        print("Num GPUs Available:", len(gpus))
        for gpu in gpus:
            print(f"Device name: {gpu.name}")
            print(f"Device details: {gpu}")

        # Try to set memory growth
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Memory growth set for GPU: {gpu.name}")
        except RuntimeError as e:
            print(e)

    # Test TensorFlow GPU functionality
    try:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            c = tf.matmul(a, b)
            print("Matrix multiplication result (should be on GPU):")
            print(c)
    except RuntimeError as e:
        print(e)

if __name__ == "__main__":
    check_gpu()
