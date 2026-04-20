# TOY-91 sample dataset

This is a small dataset for testing object detection. It contains 91 images of various toys, with only 3 toys defined as target objects and their class names defined, along with reference images.

`exemplar.json`: Class IDs and names for the 3 classes, and the paths to their reference images.

`refer_images/*.jpg`: Reference images for the 3 classes. Each class has at least one reference image.

`query_images/*.jpg`: 91 640x480 RGB JPEG images of various toys.

`query_images/*.gt.json`: Ground truth of the detected objects in the 91 images.
