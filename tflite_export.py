import tensorflow as tf
# make a converter object from the saved tensorflow file
converter = tf.lite.TFLiteConverter.from_saved_model('pixellink.pb', #TensorFlow freezegraph .pb model file
                                                      #input_arrays=['input.1'], # name of input arrays as defined in torch.onnx.export function before.
                                                      #output_arrays=['258', '354']  # name of output arrays defined in torch.onnx.export function before.
                                                      )
# tell converter which type of optimization techniques to use
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# to view the best option for optimization read documentation of tflite about optimization
# go to this link https://www.tensorflow.org/lite/guide/get_started#4_optimize_your_model_optional

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]

# convert the model
print('TFlite model converting')
tf_lite_model = converter.convert()
# save the converted model
open('pixellink.tflite', 'wb').write(tf_lite_model)
print('TFlite model saved')