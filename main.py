import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import motion
import CNN_img
import tensorflow_compression as tfc 
from MC_network import MC
import MC_network
import os

os.makedirs('./bitstream', exist_ok=True)




frame1_path = r"C:\Users\anish\OneDrive\Desktop\motion estimation\extracted_frames\frame_0000.jpg"  # Replace with the path to the first frame
frame2_path = r"C:\Users\anish\OneDrive\Desktop\motion estimation\extracted_frames\frame_0001.jpg"  # Replace with the path to the second frame

frame1 = cv2.imread(frame1_path)
frame2 = cv2.imread(frame2_path)

Height, Width, _ = frame1.shape

# Normalize and expand dimensions for batch processing
frame1 = frame1 / 255.0
frame2 = frame2 / 255.0
frame1 = np.expand_dims(frame1, axis=0)
frame2 = np.expand_dims(frame2, axis=0)

batch_size = 1
Channel = 3

# Define placeholders
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




######
# warping the frame
Y1_warp = tf.contrib.image.dense_image_warp(Y0_com, flow_hat)








# motion compensation
MC_input = tf.concat([flow_hat, Y0_com, Y1_warp], axis=-1)
Y1_MC = MC_network.MC(MC_input)








# Residual
Res = Y1_raw - Y1_MC





# Residual Encoding
res_latent = CNN_img.Res_analysis(Res, num_filters=128, M=128)

entropy_bottleneck_res = tfc.EntropyBottleneck()
string_res = entropy_bottleneck_res.compress(res_latent)
string_res = tf.squeeze(string_res, axis=0)

res_latent_hat, Res_likelihoods = entropy_bottleneck_res(res_latent, training=False)

Res_hat = CNN_img.Res_synthesis(res_latent_hat, num_filters=128)









################################################################

# Reconstructed frame
Y1_com = tf.clip_by_value(Res_hat + Y1_MC, 0, 1)




################################################################







# Start a TensorFlow session
saver = tf.train.Saver(max_to_keep=None)
model_path = './OpenDVC_model/PSNR_1024_model/model.ckpt'  # Update this path as needed







###################################################
#----------------------Tensorflow Session----------------------
###################################################
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, save_path=model_path)

    # Compute optical flow
    flow = sess.run(flow_hat, feed_dict={Y0_com: frame1, Y1_raw: frame2})[0]  # Remove batch dimension

    # Compute motion-compensated frame (Y1_warp)
    y1_warp_output = sess.run(Y1_warp, feed_dict={
        Y0_com: frame1,
        Y1_raw: frame2
    })[0]  # Remove batch dimension


    # Compute motion-compensated frame from MC network
    y1_mc_output = sess.run(Y1_MC, feed_dict={
        Y0_com: frame1,
        Y1_raw: frame2
    })[0]  # Remove batch dimension



    # Run residual computation inside session
    res_output = sess.run(Res, feed_dict={
        Y0_com: frame1,
        Y1_raw: frame2
    })[0]  # Remove batch dimension




    # Get reconstructed residual from entropy bottleneck
    res_hat_output = sess.run(Res_hat, feed_dict={
        Y0_com: frame1,
        Y1_raw: frame2
    })[0]  # Remove batch dimension





    # Final reconstructed frame after residual add-back
    y1_com_output = sess.run(Y1_com, feed_dict={
        Y0_com: frame1,
        Y1_raw: frame2
    })[0]  # Remove batch dimension

    

    # Bitstream
    
    # Compute and save the bitstrings
    mv_bitstring, res_bitstring = sess.run([string_mv, string_res], 
                                          feed_dict={Y0_com: frame1, Y1_raw: frame2})
    
    # Save the bitstrings to files
    with open('./bitstream/mv_bitstream.bin', 'wb') as f:
        f.write(mv_bitstring)
    with open('./bitstream/res_bitstream.bin', 'wb') as f:
        f.write(res_bitstring)
    
    print(f"Motion vector bitstream size: {len(mv_bitstring)} bytes")
    print(f"Residual bitstream size: {len(res_bitstring)} bytes")

###################################################
#----------------------Visualization----------------------
###################################################


plt.imshow(y1_com_output.squeeze() if y1_com_output.shape[-1] == 1 else y1_com_output)
plt.title("Y1_com (Final Reconstructed Frame)")
plt.axis('off')
plt.show()




# Visualize the final motion-compensated frame
is_exact = np.array_equal(frame2[0], y1_mc_output)
print("Residual is exactly zero:", is_exact)

# Clip values to [0,1] range for safe visualization
y1_mc_output = np.clip(y1_mc_output, 0.0, 1.0)

if y1_mc_output.shape[-1] == 1:
    plt.imshow(y1_mc_output.squeeze(), cmap='gray')
else:
    plt.imshow(y1_mc_output)
plt.title("Y1_MC (Refined Motion Compensation)")
plt.axis('off')



abs_res = np.abs(res_output)
abs_res = np.clip(abs_res, 0.0, 1.0)

plt.imshow(abs_res.squeeze(), cmap='gray')
plt.title("Residual Magnitude |Y1_raw - Y1_MC|")
plt.axis('off')
plt.show()







# Warped frame
    # Clip values to [0,1] for display
y1_warp_output = np.clip(y1_warp_output, 0.0, 1.0)

if y1_warp_output.shape[-1] == 1:
    plt.imshow(y1_warp_output.squeeze(), cmap='gray')
else:
    plt.imshow(y1_warp_output)
plt.title("Y1_warp (Motion Compensated Frame)")
plt.axis('off')













# Visualize the optical flow 
h, w, _ = flow.shape
y, x = np.mgrid[0:h:10, 0:w:10].reshape(2, -1).astype(int)
fx, fy = flow[y, x].T

scale_factor = 10.0  
fx_scaled = fx * scale_factor
fy_scaled = fy * scale_factor
frame1_uint8 = (frame1[0] * 255).astype(np.uint8)

plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(frame1_uint8, cv2.COLOR_BGR2RGB))
plt.quiver(x, y, fx_scaled, fy_scaled, color='r', angles='xy', scale_units='xy', scale=1)
plt.title("Optical Flow Visualization")











# Flow heatmap
flow_magnitude = np.sqrt(np.sum(np.square(flow), axis=-1))
norm_mag = cv2.normalize(flow_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
heatmap = cv2.applyColorMap(norm_mag, cv2.COLORMAP_JET)
frame1_uint8 = (frame1[0] * 255).astype(np.uint8)
overlay = cv2.addWeighted(frame1_uint8, 0.6, heatmap, 0.4, 0)

plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.title("Motion Flow Heatmap Overlay")
plt.axis("off")
plt.show()
