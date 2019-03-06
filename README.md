# Gradient Penalty DCGAN
Gradient Penalty DCGAN built on Keras.

gp-gan.py uses DCGAN architecture.
gp-gan-res.py uses Residual Blocks with DCGAN.
ggr-alpha.py uses AlphaGAN architecture with Residual Blocks.

Data collected from here:
https://www.reddit.com/r/flowers

Related Papers:
"Which Training Methods for GANs do actually Converge?"
https://arxiv.org/abs/1801.04406

"Variational Approaches for Auto-Encoding Generative Adversarial Networks"
https://arxiv.org/abs/1706.04987




![alt text](https://i.imgur.com/2ajHhqV.jpg)

# Use
To use, rename all images in your dataset according to this convention: "im (n).png".
On windows you can do this by selecting all images, and renaming the first to "im".
Then put all images into a folder under /data/.
Change gp-gan-res.py to fit what you want (image size, directory, etc.)
Run gp-gan-res.py.
