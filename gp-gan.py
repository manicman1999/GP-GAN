# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 13:57:15 2018

@author: mtm916
"""


#import numpy as np
from PIL import Image
from math import floor
import numpy as np
import time
from functools import partial

im_size = 32
latent_size = 64
BATCH_SIZE = 8
directory = "Swords"

def noise(n):
    return np.random.normal(0.0, 1.0, size = [n, latent_size])


#Get random samples from an array
def get_rand(array, amount):
    
    idx = np.random.randint(0, array.shape[0], amount)
    return array[idx]

#Import Images Function
def import_images(loc, flip = True, suffix = 'png'):
    
    out = []
    cont = True
    i = 1
    print("Importing Images...")
    
    while(cont):
        try:
            temp = Image.open("data/"+loc+"/im ("+str(i)+")."+suffix+"").convert('RGB')
            temp = temp.resize((im_size, im_size), Image.BICUBIC)
            temp1 = np.array(temp.convert('RGB'), dtype='float32') / 255
            out.append(temp1)
            if flip:
                out.append(np.flip(out[-1], 1))
            
            i = i + 1
        except:
            cont = False
        
    print(str(i-1) + " images imported.")
            
    return np.array(out)

    
from keras.layers import Conv2D, BatchNormalization, Dense, AveragePooling2D, Conv2DTranspose
from keras.layers import Reshape, UpSampling2D, Activation, Dropout, Flatten, Input
from keras.models import model_from_json, Sequential
from keras.optimizers import RMSprop
import keras.backend as K
from keras.models import Model

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


def gradient_penalty_loss(y_true, y_pred, averaged_samples, weight):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradient_penalty = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    
    # weight * ||grad||^2
    # Penalize the gradient norm
    return K.mean(gradient_penalty * weight)


def g_block(f, b = True):
    #Upsample, Convolution, BatchNorm, Activation
    temp = Sequential()
    temp.add(UpSampling2D())
    temp.add(Conv2D(filters = f, kernel_size = 3, padding = 'same', kernel_initializer = 'he_uniform'))
    if b:
        temp.add(BatchNormalization())
    temp.add(Activation('relu'))
    
    return temp

def d_block(f, p = True):
    #Convolution, Activation, Pool
    temp = Sequential()
    temp.add(Conv2D(filters = f, kernel_size = 3, padding = 'same', kernel_initializer = 'he_uniform'))
    temp.add(Activation('relu'))
    if p:
        temp.add(AveragePooling2D())
        
    return temp
    
class GAN(object):
    
    def __init__(self, lr = 0.0001):
        
        #Models
        self.D = None
        self.G = None
        
        self.DM = None
        self.AM = None
        
        #Config
        self.LR = lr
        self.steps = 1
        
        #Init Models
        self.discriminator()
        self.generator()
        
    def discriminator(self):
        
        if self.D:
            return self.D
        
        self.D = Sequential()
        
        self.D.add(Activation('linear', input_shape = [im_size, im_size, 3]))
        
        # Size
        self.D.add(d_block(8))
        self.D.add(d_block(16))
        self.D.add(d_block(32))
        
        #Size / 8
        if (im_size >= 32):
            self.D.add(d_block(64))
        
        # Size / 16
        if (im_size >= 64):
            self.D.add(d_block(128))
        
        # Size / 32
        if (im_size >= 128):
            self.D.add(d_block(256))
        
        # Size / 64
        if (im_size >= 256):
            self.D.add(d_block(512))
            
        # Size / 128
        if (im_size >= 512):
            self.D.add(d_block(1024))
            
        self.D.add(d_block(im_size, p = False))
            
        self.D.add(Flatten())
        
        self.D.add(Dense(128, activation = 'relu'))
        
        self.D.add(Dropout(0.6))
        self.D.add(Dense(1, activation = 'sigmoid'))
        
        return self.D
    
    def generator(self):
        
        if self.G:
            return self.G
        
        self.G = Sequential()
        
        self.G.add(Reshape(target_shape = [1, 1, latent_size], input_shape = [latent_size]))
        
        #4x4
        self.G.add(Conv2DTranspose(im_size, kernel_size = 4, activation = 'relu', kernel_initializer = 'he_uniform'))
        
        if(im_size >= 512):
            self.G.add(g_block(1024)) # Size / 64
        if(im_size >= 256):
            self.G.add(g_block(256)) # Size / 32
        if(im_size >= 128):
            self.G.add(g_block(128)) # Size / 16
        if(im_size >= 64):
            self.G.add(g_block(64)) # Size / 8
        if(im_size >= 32):
            self.G.add(g_block(32)) # Size / 4
        
        self.G.add(g_block(16)) # Size / 2
        self.G.add(g_block(8, b = False)) # Size
        
        self.G.add(Conv2D(filters = 3, kernel_size = 1, padding = 'same', activation = 'sigmoid'))
        
        return self.G
    
    def AdModel(self):
        
        #D does not update
        self.D.trainable = False
        for layer in self.D.layers:
            layer.trainable = False
        
        #G does update
        self.G.trainable = True
        for layer in self.G.layers:
            layer.trainable = True
        
        #This model is simple sequential
        if self.AM == None:
            self.AM = Sequential()
            self.AM.add(self.G)
            self.AM.add(self.D)
            
        self.AM.compile(optimizer = RMSprop(self.LR), loss = 'binary_crossentropy')
        
        return self.AM
    
    def DisModel(self):
        
        #D does update
        self.D.trainable = True
        for layer in self.D.layers:
            layer.trainable = True
        
        #G does not update
        self.G.trainable = False
        for layer in self.G.layers:
            layer.trainable = False
        
        # Real Pipeline
        ri = Input(shape = [im_size, im_size, 3])
        dr = self.D(ri)
        
        # Fake Pipeline
        gi = Input(shape = [latent_size])
        gf = self.G(gi)
        df = self.D(gf)
        
        # Samples for gradient penalty
        # For r1 use real samples (ri)
        # For r2 use fake samples (gf)
        da = self.D(ri)
        
        # Model With Inputs and Outputs
        self.DM = Model(inputs=[ri, gi], outputs=[dr, df, da])
        
        # Create partial of gradient penalty loss
        # For r1, averaged_samples = ri
        # For r2, averaged_samples = gf
        # Weight of 10 typically works
        partial_gp_loss = partial(gradient_penalty_loss, averaged_samples = ri, weight = 50)
        
        #Compile With Corresponding Loss Functions
        self.DM.compile(optimizer=RMSprop(self.LR), loss=['binary_crossentropy', 'binary_crossentropy', partial_gp_loss])
        
        return self.DM
        
        


class WGAN(object):
    
    def __init__(self, steps = -1, lr = 0.0001, silent = True):
        
        self.GAN = GAN(lr = lr)
        self.DisModel = self.GAN.DisModel()
        self.AdModel = self.GAN.AdModel()
        self.generator = self.GAN.generator()
        
        if steps >= 0:
            self.GAN.steps = steps
        
        self.lastblip = time.clock()
        
        self.noise_level = 0
        
        self.ImagesA = import_images(directory)
        
        self.silent = silent
        
        self.ones = np.ones((BATCH_SIZE, 1), dtype=np.float32) - np.random.uniform(0.0, 0.01, size = (BATCH_SIZE, 1))
        self.zeros = np.zeros((BATCH_SIZE, 1), dtype=np.float32) + np.random.uniform(0.0, 0.01, size = (BATCH_SIZE, 1))
        
        self.enoise = noise(32)
        
    def train(self):
        
        #Train Alternating
        a = self.train_dis()
        b = self.train_gen()
        
        #Print info
        if self.GAN.steps % 20 == 0 and not self.silent:
            print("\n\nRound " + str(self.GAN.steps) + ":")
            print("D: " + str(a))
            print("G: " + str(b))
            s = round((time.clock() - self.lastblip) * 1000) / 1000
            print("T: " + str(s) + " sec")
            self.lastblip = time.clock()
        
        #Save Model
        if self.GAN.steps % 500 == 0:
            self.save(floor(self.GAN.steps / 10000))
        
        self.GAN.steps = self.GAN.steps + 1
        
    def train_dis(self):
        
        #Get Data
        train_data = [get_rand(self.ImagesA, BATCH_SIZE), noise(BATCH_SIZE)]
        
        #Train
        d_loss = self.DisModel.train_on_batch(train_data, [self.ones, self.zeros, self.zeros])
        
        return d_loss
        
    def train_gen(self):
        
        #Train
        g_loss = self.AdModel.train_on_batch(noise(BATCH_SIZE), self.ones)
        
        return g_loss
    
    def evaluate(self, num = 0, trunc = 2.0):
        
        n2 = noise(32)
        
        im2 = self.generator.predict(n2)
        im3 = self.generator.predict(self.enoise)
        
        r12 = np.concatenate(im2[:8], axis = 1)
        r22 = np.concatenate(im2[8:16], axis = 1)
        r32 = np.concatenate(im2[16:24], axis = 1)
        r42 = np.concatenate(im2[24:32], axis = 1)
        r13 = np.concatenate(im3[:8], axis = 1)
        r23 = np.concatenate(im3[8:16], axis = 1)
        r33 = np.concatenate(im3[16:24], axis = 1)
        r43 = np.concatenate(im3[24:32], axis = 1)
        
        c1 = np.concatenate([r12, r22, r32, r42, r13, r23, r33, r43], axis = 0)
        
        x = Image.fromarray(np.uint8(c1*255))
        
        x.save("Results/i"+str(num)+".png")
    
    def saveModel(self, model, name, num):
        json = model.to_json()
        with open("Models/"+name+".json", "w") as json_file:
            json_file.write(json)
            
        model.save_weights("Models/"+name+"_"+str(num)+".h5")
        
    def loadModel(self, name, num):
        
        file = open("Models/"+name+".json", 'r')
        json = file.read()
        file.close()
        
        mod = model_from_json(json)
        mod.load_weights("Models/"+name+"_"+str(num)+".h5")
        
        return mod
    
    def save(self, num): #Save JSON and Weights into /Models/
        self.saveModel(self.GAN.G, "gen", num)
        self.saveModel(self.GAN.D, "dis", num)
        

    def load(self, num): #Load JSON and Weights from /Models/
        steps1 = self.GAN.steps
        
        self.GAN = None
        self.GAN = GAN()

        #Load Models
        self.GAN.G = self.loadModel("gen", num)
        self.GAN.D = self.loadModel("dis", num)
        
        self.GAN.steps = steps1
        
        self.generator = self.GAN.generator()
        self.DisModel = self.GAN.DisModel()
        self.AdModel = self.GAN.AdModel()
    
        
    def sample(self, n):
        
        return self.generator.predict(noise(n))
    
    def instance_noise(self):
        
        self.AmagesA = np.array(self.AmagesA)
        
        self.ImagesA = self.AmagesA + np.random.uniform(-self.noise_level, self.noise_level, size = self.AmagesA.shape)
        
        
        
        
if __name__ == "__main__":
    model = WGAN(499, lr = 0.0002, silent = False)
    
    while(model.GAN.steps < 5000000):
        
        #model.eval()
        model.train()
        
        if model.GAN.steps % 1000 == 0:
            model.evaluate(int(model.GAN.steps / 1000))


