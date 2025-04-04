import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import motion
import CNN_img
import tensorflow_compression as tfc 

# Load two frames
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
flow_latent_hat, _ = entropy_bottleneck_mv(flow_latent, training=False)

flow_hat = CNN_img.MV_synthesis(flow_latent_hat, num_filters=128)

# Start a TensorFlow session
saver = tf.train.Saver(max_to_keep=None)
model_path = './OpenDVC_model/PSNR_1024_model/model.ckpt'  # Update this path as needed

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, save_path=model_path)

    # Compute optical flow
    flow = sess.run(flow_hat, feed_dict={Y0_com: frame1, Y1_raw: frame2})[0]  # Remove batch dimension

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
