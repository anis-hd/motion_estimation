import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import motion
import CNN_img
import tensorflow_compression as tfc 
from MC_network import MC
import MC_network

# Create output folder for compressed bitstreams and sampled frames
output_folder = './compressed_bitstreams'
os.makedirs(output_folder, exist_ok=True)

# Directory containing the frames (assumed to be 50 frames)
frame_dir = './extracted_frames'
frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(('.jpg', '.png'))])
if len(frame_files) < 50:
    raise ValueError("Expected at least 50 frames in the folder.")

# Get frame dimensions from the first frame (assuming all frames are the same size)
first_frame_path = os.path.join(frame_dir, frame_files[0])
frame_temp = cv2.imread(first_frame_path)
Height, Width, Channel = frame_temp.shape

# Define placeholders for TensorFlow graph
batch_size = 1
Y0_com = tf.placeholder(tf.float32, [batch_size, Height, Width, Channel])
Y1_raw = tf.placeholder(tf.float32, [batch_size, Height, Width, Channel])

# Motion estimation using the optical flow network
with tf.variable_scope("flow_motion"):
    flow_tensor, _, _, _, _, _ = motion.optical_flow(Y0_com, Y1_raw, batch_size, Height, Width)

# Encode flow
flow_latent = CNN_img.MV_analysis(flow_tensor, num_filters=128, M=128)
entropy_bottleneck_mv = tfc.EntropyBottleneck()
string_mv = entropy_bottleneck_mv.compress(flow_latent)
string_mv = tf.squeeze(string_mv, axis=0)
flow_latent_hat, MV_likelihoods = entropy_bottleneck_mv(flow_latent, training=False)
flow_hat = CNN_img.MV_synthesis(flow_latent_hat, num_filters=128)

# Warping the frame
Y1_warp = tf.contrib.image.dense_image_warp(Y0_com, flow_hat)

# Motion compensation
MC_input = tf.concat([flow_hat, Y0_com, Y1_warp], axis=-1)
Y1_MC = MC_network.MC(MC_input)

# Residual computation
Res = Y1_raw - Y1_MC

# Residual Encoding
res_latent = CNN_img.Res_analysis(Res, num_filters=128, M=128)
entropy_bottleneck_res = tfc.EntropyBottleneck()
string_res = entropy_bottleneck_res.compress(res_latent)
string_res = tf.squeeze(string_res, axis=0)
res_latent_hat, Res_likelihoods = entropy_bottleneck_res(res_latent, training=False)
Res_hat = CNN_img.Res_synthesis(res_latent_hat, num_filters=128)

# Reconstructed frame
Y1_computed = tf.clip_by_value(Res_hat + Y1_MC, 0, 1)

# Saver and model path
saver = tf.train.Saver(max_to_keep=None)
model_path = './OpenDVC_model/PSNR_1024_model/model.ckpt'  # Update this path if needed

# Start a TensorFlow session (load the model once, then loop through frame pairs)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, save_path=model_path)

    # Loop over the frames in steps of 2 (two frames at a time)
    for i in range(0, 50, 2):
        # Construct file paths for the two frames
        frame1_file = os.path.join(frame_dir, frame_files[i])
        frame2_file = os.path.join(frame_dir, frame_files[i+1])
        
        # Read frames using cv2 and normalize (so they’re between 0 and 1)
        frame1 = cv2.imread(frame1_file)
        frame2 = cv2.imread(frame2_file)
        if frame1 is None or frame2 is None:
            print(f"Skipping pair ({frame1_file}, {frame2_file}) because one could not be read.")
            continue

        frame1 = frame1 / 255.0
        frame2 = frame2 / 255.0
        frame1 = np.expand_dims(frame1, axis=0)  # Add batch dimension
        frame2 = np.expand_dims(frame2, axis=0)
        
        # Run the TensorFlow session to compute all outputs
        feed_dict = {Y0_com: frame1, Y1_raw: frame2}
        
        # You can compute intermediate outputs as needed:
        flow_output, y1_warp_output, y1_mc_output, res_output, res_hat_output, y1_com_output, mv_bitstring, res_bitstring = sess.run(
            [flow_hat, Y1_warp, Y1_MC, Res, Res_hat, Y1_computed, string_mv, string_res],
            feed_dict=feed_dict
        )
        
        # Save the bitstreams for this pair
        mv_filename = os.path.join(output_folder, f"mv_bitstream_pair_{i:04d}_{i+1:04d}.bin")
        res_filename = os.path.join(output_folder, f"res_bitstream_pair_{i:04d}_{i+1:04d}.bin")
        with open(mv_filename, 'wb') as f:
            f.write(mv_bitstring)
        with open(res_filename, 'wb') as f:
            f.write(res_bitstring)
        print(f"Processed frame pair {i} and {i+1}: MV bitstream = {len(mv_bitstring)} bytes, Residual bitstream = {len(res_bitstring)} bytes")
        
        # Every 10th frame (based on the index of the second frame), save the reconstructed frame
        if (i+1) % 10 == 0:
            # Convert the reconstructed frame back to uint8 format
            recon_frame = (y1_com_output[0] * 255).astype(np.uint8)
            frame_save_path = os.path.join(output_folder, f"reconstructed_frame_{i+1:04d}.png")
            cv2.imwrite(frame_save_path, recon_frame)
            print(f"Saved reconstructed frame at {frame_save_path}")

