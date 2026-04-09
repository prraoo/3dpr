import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import sys
# from PIL import Image
# import imageio.v3
# import avif.pillow
import cv2
import numpy as np


input_file = sys.argv[1]
image = cv2.imread(input_file, -1)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print(image.max(), image.min())
norm_image = np.zeros_like(image)
norm_image = cv2.normalize(image, norm_image, 0, 1, cv2.NORM_MINMAX)
print(norm_image.max(), norm_image.min())

clipped_image = np.array(np.clip(norm_image, 0, 1) * (255), dtype=np.uint16)
# print(clipped_image.max(), clipped_image.min(), clipped_image.shape, clipped_image.dtype)
#
# pil_image = Image.fromarray(clipped_image)
#
# avif.encode(
#     "output.avif",
#     clipped_image,
#     depth=10,     # 8, 10, or 12
#     quality=90,    # 0-100 (higher = better)
#     speed=6,       # 0-10 (lower = slower + better compression)
# )

# Save as AVIF
cv2.imwrite('output.avif', clipped_image)
###########################
# import av
# import imageio
# import numpy as np

# # Read the EXR file
# img = imageio.v3.imread(input_file)
#
# # Clip and scale to 12-bit range (0-4095)
# img_clipped = np.array(np.round(np.interp(img, [0, 1], [0, 4095])), dtype='uint16')
#
# # Create an av frame from the NumPy array in RGB format (48-bit RGB)
# frame = av.Frame.from_ndarray(img_clipped, format='rgb48le')
#
# # Reformat to YUV420P12LE for AV1 encoding
# frame = frame.reformat('yuv420p12le')
#
# # Create an AVIF container
# output_container = av.open('output.avif', 'w')
#
# # Add a stream with libaom codec
# stream = output_container.add_stream('libaom', rate=1)
# stream.width = frame.width
# stream.height = frame.height
# stream.pix_fmt = 'yuv420p12le'
#
# # Set compression quality using CRF (e.g., 30 for medium quality)
# crf_value = 30  # Adjust this (10 for high quality, 50 for high compression)
# stream.options = {'crf': str(crf_value)}
#
# # Mux the frame into the container
# output_container.mux(frame)
#
# # Close the container
# output_container.close()
#
# print(f"Converted input.exr to output.avif with CRF {crf_value}")

