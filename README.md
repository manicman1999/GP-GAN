# Gradient Penalty DCGAN
Gradient Penalty DCGAN built on Keras.

Data collected from here:
https://www.reddit.com/r/flowers

Related Paper:
"Which Training Methods for GANs do actually Converge?"
https://arxiv.org/pdf/1801.04406.pdf


![alt text](https://i.imgur.com/fcujEbp.jpg)

# Use
To use, rename all images in your dataset according to this convention: "im (n).png".
On windows you can do this by selecting all images, and renaming the first to "im".
Then put all images into a folder under /data/.
Change gp-gan.py to fit what you want (image size, directory, etc.)
Run gp-gan.py.
