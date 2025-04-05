import tensorflow as tf
import numpy as np
import cv2
import os
import tensorflow_compression as tfc

# Load the first frame
frame1 = np.load('./bitstream/frame1.npy')
frame1 = np.expand_dims(frame1, axis=0)  # Add batch dimension
Height, Width, Channel = frame1.shape[1:]

# Define placeholders
Y0_com = tf.placeholder(tf.float32, [1, Height, Width, Channel])

# Create entropy bottlenecks
entropy_bottleneck_mv = tfc.EntropyBottleneck(dtype=tf.float32)
entropy_bottleneck_res = tfc.EntropyBottleneck(dtype=tf.float32)

# Read the bitstreams
with open('./bitstream/mv_bitstream.bin', 'rb') as f:
    mv_bitstring = f.read()
with open('./bitstream/res_bitstream.bin', 'rb') as f:
    res_bitstring = f.read()

# Convert to tensors
string_mv = tf.expand_dims(mv_bitstring, 0)
string_res = tf.expand_dims(res_bitstring, 0)

# Define the latent shapes
flow_latent_shape = (Height//16, Width//16, 128)
res_latent_shape = (Height//16, Width//16, 128)

# Decompress motion vectors
flow_latent_hat = entropy_bottleneck_mv.decompress(
    string_mv, flow_latent_shape, channels=128)

# Create MV synthesis network with proper variable scoping
with tf.variable_scope("MV_synthesis"):
    with tf.variable_scope("layer_0"):
        layer = tfc.SignalConv2D(
            128, (3, 3), corr=False, strides_up=2, padding="same_zeros",
            use_bias=True, activation=tfc.GDN(inverse=True))
        flow_hat = layer(flow_latent_hat)
    
    with tf.variable_scope("layer_1"):
        layer = tfc.SignalConv2D(
            128, (3, 3), corr=False, strides_up=2, padding="same_zeros",
            use_bias=True, activation=tfc.GDN(inverse=True))
        flow_hat = layer(flow_hat)
    
    with tf.variable_scope("layer_2"):
        layer = tfc.SignalConv2D(
            128, (3, 3), corr=False, strides_up=2, padding="same_zeros",
            use_bias=True, activation=tfc.GDN(inverse=True))
        flow_hat = layer(flow_hat)
    
    with tf.variable_scope("layer_3"):
        layer = tfc.SignalConv2D(
            2, (3, 3), corr=False, strides_up=2, padding="same_zeros",
            use_bias=True, activation=None)
        flow_hat = layer(flow_hat)

# Warping
Y1_warp = tf.contrib.image.dense_image_warp(Y0_com, flow_hat)

# Motion compensation (simplified)
Y1_MC = Y1_warp  # Replace with your actual MC network if needed

# Decompress residual
res_latent_hat = entropy_bottleneck_res.decompress(
    string_res, res_latent_shape, channels=128)

# Create Res synthesis network with proper variable scoping
with tf.variable_scope("synthesis"):  # Note: matches Res_synthesis scope in CNN_img.py
    with tf.variable_scope("layer_0"):
        layer = tfc.SignalConv2D(
            128, (5, 5), corr=False, strides_up=2, padding="same_zeros",
            use_bias=True, activation=tfc.GDN(inverse=True))
        Res_hat = layer(res_latent_hat)
    
    with tf.variable_scope("layer_1"):
        layer = tfc.SignalConv2D(
            128, (5, 5), corr=False, strides_up=2, padding="same_zeros",
            use_bias=True, activation=tfc.GDN(inverse=True))
        Res_hat = layer(Res_hat)
    
    with tf.variable_scope("layer_2"):
        layer = tfc.SignalConv2D(
            128, (5, 5), corr=False, strides_up=2, padding="same_zeros",
            use_bias=True, activation=tfc.GDN(inverse=True))
        Res_hat = layer(Res_hat)
    
    with tf.variable_scope("layer_3"):
        layer = tfc.SignalConv2D(
            3, (5, 5), corr=False, strides_up=2, padding="same_zeros",
            use_bias=True, activation=None)
        Res_hat = layer(Res_hat)

# Reconstruct final frame
Y1_com = tf.clip_by_value(Res_hat + Y1_MC, 0, 1)

# Get the list of variables we expect to restore
# (Exclude the entropy bottleneck variables since they're initialized separately)
model_vars = [
    v for v in tf.global_variables() 
    if 'MV_synthesis' in v.name or 'synthesis' in v.name
]

# Create a saver that only restores the model variables
model_saver = tf.train.Saver(var_list=model_vars)
model_path = './OpenDVC_model/PSNR_1024_model/model.ckpt'

with tf.Session() as sess:
    # First initialize all variables
    sess.run(tf.global_variables_initializer())
    
    # Initialize entropy bottlenecks
    sess.run(entropy_bottleneck_mv.init_ops())
    sess.run(entropy_bottleneck_res.init_ops())
    
    # Then restore only the model variables
    model_saver.restore(sess, save_path=model_path)
    
    # Reconstruct the frame
    reconstructed_frame = sess.run(Y1_com, feed_dict={Y0_com: frame1})[0]
    
    # Convert to 8-bit for display/saving
    reconstructed_frame_uint8 = (reconstructed_frame * 255).astype(np.uint8)
    
    # Save the reconstructed frame
    cv2.imwrite('./bitstream/reconstructed_frame.jpg', reconstructed_frame_uint8)
    print("Successfully reconstructed and saved the second frame")
    
    # Display
    cv2.imshow('Reconstructed Frame', cv2.cvtColor(reconstructed_frame_uint8, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
    cv2.destroyAllWindows()