import tensorflow as tf
import numpy as np
import cv2
import os
import motion
import CNN_img
import tensorflow_compression as tfc 
from MC_network import MC
import MC_network

# Configuration
os.makedirs('./bitstream', exist_ok=True)

# Load frames
frame1_path = r"C:\Users\anish\OneDrive\Desktop\motion estimation\extracted_frames\frame_0000.jpg"  # Replace with your path
frame2_path = r"C:\Users\anish\OneDrive\Desktop\motion estimation\extracted_frames\frame_0001.jpg"  # Replace with your path

frame1 = cv2.imread(frame1_path)
frame2 = cv2.imread(frame2_path)

Height, Width, _ = frame1.shape

# Normalize and expand dimensions
frame1 = frame1 / 255.0
frame2 = frame2 / 255.0
frame1 = np.expand_dims(frame1, axis=0)
frame2 = np.expand_dims(frame2, axis=0)

batch_size = 1
Channel = 3

# Define placeholders
Y0_com = tf.placeholder(tf.float32, [batch_size, Height, Width, Channel])
Y1_raw = tf.placeholder(tf.float32, [batch_size, Height, Width, Channel])

# Motion estimation
with tf.variable_scope("flow_motion"):
    flow_tensor, _, _, _, _, _ = motion.optical_flow(Y0_com, Y1_raw, batch_size, Height, Width)

# Encode flow
flow_latent = CNN_img.MV_analysis(flow_tensor, num_filters=128, M=128)
entropy_bottleneck_mv = tfc.EntropyBottleneck()
string_mv = entropy_bottleneck_mv.compress(flow_latent)
string_mv = tf.squeeze(string_mv, axis=0)
flow_latent_hat, _ = entropy_bottleneck_mv(flow_latent, training=False)
flow_hat = CNN_img.MV_synthesis(flow_latent_hat, num_filters=128)

# Warping and motion compensation
Y1_warp = tf.contrib.image.dense_image_warp(Y0_com, flow_hat)
MC_input = tf.concat([flow_hat, Y0_com, Y1_warp], axis=-1)
Y1_MC = MC_network.MC(MC_input)

# Residual
Res = Y1_raw - Y1_MC

# Residual Encoding
res_latent = CNN_img.Res_analysis(Res, num_filters=128, M=128)
entropy_bottleneck_res = tfc.EntropyBottleneck()
string_res = entropy_bottleneck_res.compress(res_latent)
string_res = tf.squeeze(string_res, axis=0)
res_latent_hat, _ = entropy_bottleneck_res(res_latent, training=False)
Res_hat = CNN_img.Res_synthesis(res_latent_hat, num_filters=128)

# Reconstructed frame
Y1_com = tf.clip_by_value(Res_hat + Y1_MC, 0, 1)

# Run the session
saver = tf.train.Saver(max_to_keep=None)
model_path = './OpenDVC_model/PSNR_1024_model/model.ckpt'

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, save_path=model_path)
    
    # Save the bitstreams
    mv_bitstring, res_bitstring = sess.run([string_mv, string_res], 
                                         feed_dict={Y0_com: frame1, Y1_raw: frame2})
    
    with open('./bitstream/mv_bitstream.bin', 'wb') as f:
        f.write(mv_bitstring)
    with open('./bitstream/res_bitstream.bin', 'wb') as f:
        f.write(res_bitstring)
    
    print(f"Saved motion vector bitstream: {len(mv_bitstring)} bytes")
    print(f"Saved residual bitstream: {len(res_bitstring)} bytes")
    
    # Also save the first frame for reconstruction
    np.save('./bitstream/frame1.npy', frame1[0])  # Save without batch dimension
    print("Saved first frame for reconstruction")