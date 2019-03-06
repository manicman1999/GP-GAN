# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 14:13:08 2019

@author: Matthew
"""

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


    
from keras.layers import Conv2D, Dense, AveragePooling2D, Conv2DTranspose, BatchNormalization
from keras.layers import Reshape, UpSampling2D, Activation, Dropout, Flatten, Input, add
from keras.models import model_from_json, Sequential
from keras.optimizers import RMSprop
import keras.backend as K
from keras.models import Model


def gradient_penalty_loss(y_true, y_pred, averaged_samples, weight):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradient_penalty = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    
    # weight * ||grad||^2
    # Penalize the gradient norm
    return K.mean(gradient_penalty * weight)


def g_block(inp, fil, b = True):
    route1 = UpSampling2D()(inp)
    route1 = Dense(fil, kernel_initializer = 'he_uniform', use_bias = False)(route1)
    
    route2 = UpSampling2D()(inp)
    route2 = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_uniform')(route2)
    if(b):
        route2 = BatchNormalization()(route2)
    route2 = Activation('relu')(route2)
    route2 = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_uniform')(route2)
    if(b):
        route2 = BatchNormalization()(route2)
    route2 = Activation('relu')(route2)
    
    
    out = add([route1, route2])
    
    return out

def d_block(inp, fil, p = True):
    route1 = Dense(fil, kernel_initializer = 'he_uniform', use_bias = False)(inp)
    if p:
        route1 = AveragePooling2D()(route1)
    
    route2 = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_uniform')(inp)
    route2 = Activation('relu')(route2)
    if p:
        route2 = AveragePooling2D()(route2)
    route2 = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_uniform')(route2)
    route2 = Activation('relu')(route2)
    
    out = add([route1, route2])
    
    return out
    
class GAN(object):
    
    def __init__(self, lr = 0.0001):
        
        #Models
        self.D = None
        self.G = None
        self.E = None
        
        self.DM = None
        self.AM = None
        self.VM = None
        
        #Config
        self.LR = lr
        self.steps = 1
        
        #Init Models
        self.discriminator()
        self.generator()
        self.encoder()
        
    def discriminator(self):
        
        if self.D:
            return self.D
        
        inp = Input(shape = [im_size, im_size, 3])
        
        # Size
        x = d_block(inp, 8) #Size / 2
        x = d_block(x, 16) #Size / 4
        x = d_block(x, 32) #Size / 8
        
        if (im_size > 32):
            x = d_block(x, 64) #Size / 16
        
        if (im_size > 64):
            x = d_block(x, 128) #Size / 32
        
        if (im_size > 128):
            x = d_block(x, 256) #Size / 64
        
        if (im_size > 256):
            x = d_block(x, 512) #Size / 128
            
        if (im_size > 512):
            x = d_block(x, 1024) #Size / 256
            
            
        x = Flatten()(x)
        
        x = Dense(128, activation = 'relu')(x)
        
        x = Dropout(0.6)(x)
        x = Dense(1, activation = 'sigmoid')(x)
        
        self.D = Model(inputs = inp, outputs = x)
        
        return self.D
    
    def generator(self):
        
        if self.G:
            return self.G
        
        inp = Input(shape = [latent_size])
        
        x = Reshape(target_shape = [1, 1, latent_size], input_shape = [latent_size])(inp)
        x = Conv2DTranspose(im_size, kernel_size = 4, activation = 'relu', kernel_initializer = 'he_uniform')(x)
        
        if(im_size >= 1024):
            x = g_block(x, 512) # Size / 64
        if(im_size >= 512):
            x = g_block(x, 256) # Size / 64
        if(im_size >= 256):
            x = g_block(x, 192) # Size / 32
        if(im_size >= 128):
            x = g_block(x, 128) # Size / 16
        if(im_size >= 64):
            x = g_block(x, 64) # Size / 8
            
        x = g_block(x, 32) # Size / 4
        x = g_block(x, 16) # Size / 2
        x = g_block(x, 8, b = False) # Size
        
        x = Conv2D(filters = 3, kernel_size = 1, padding = 'same', activation = 'sigmoid')(x)
        
        self.G = Model(inputs = inp, outputs = x)
        
        return self.G
    
    def encoder(self):
        
        if self.E:
            return self.E
        
        inp = Input(shape = [im_size, im_size, 3])
        
        # Size
        x = d_block(inp, 8) #Size / 2
        x = d_block(x, 16) #Size / 4
        x = d_block(x, 32) #Size / 8
        
        if (im_size > 32):
            x = d_block(x, 64) #Size / 16
        
        if (im_size > 64):
            x = d_block(x, 128) #Size / 32
        
        if (im_size > 128):
            x = d_block(x, 256) #Size / 64
        
        if (im_size > 256):
            x = d_block(x, 512) #Size / 128
            
        if (im_size > 512):
            x = d_block(x, 1024) #Size / 256
            
            
        x = Flatten()(x)
        
        x = Dense(512, activation = 'relu')(x)
        
        x = Dense(latent_size)(x)
        
        self.E = Model(inputs = inp, outputs = x)
        
        return self.E
    
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
            
        self.AM.compile(optimizer = RMSprop(self.LR, decay = 0.00001), loss = 'binary_crossentropy')
        
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
        partial_gp_loss = partial(gradient_penalty_loss, averaged_samples = ri, weight = 25)
        
        #Compile With Corresponding Loss Functions
        self.DM.compile(optimizer=RMSprop(self.LR, decay = 0.00001), loss=['binary_crossentropy', 'binary_crossentropy', partial_gp_loss])
        
        return self.DM
    
    def VAEModel(self):
        
        self.G.trainable = True
        for layer in self.G.layers:
            layer.trainable = True
        
        #VAE
        inp1 = Input(shape = [im_size, im_size, 3])
        vae = self.E(inp1)
        vae = self.G(vae)
        
        #LR
        inp2 = Input(shape = [latent_size])
        lr = self.G(inp2)
        lr = self.E(lr)
        
        self.VM = Model(inputs = [inp1, inp2], outputs = [vae, lr])
        
        self.VM.compile(optimizer=RMSprop(self.LR * 0.5, decay = 0.000015), loss=['mse', 'mse'])
        
        return self.VM
        
        


class WGAN(object):
    
    def __init__(self, steps = -1, lr = 0.0001, silent = True):
        
        self.GAN = GAN(lr = lr)
        self.DisModel = self.GAN.DisModel()
        self.AdModel = self.GAN.AdModel()
        self.VAEModel = self.GAN.VAEModel()
        self.generator = self.GAN.generator()
        
        if steps >= 0:
            self.GAN.steps = steps
        
        self.lastblip = time.clock()
        
        self.noise_level = 0
        
        self.ImagesA = import_images(directory, True)
        
        self.silent = silent
        
        self.ones = np.ones((BATCH_SIZE, 1), dtype=np.float32)
        self.zeros = np.zeros((BATCH_SIZE, 1), dtype=np.float32)
        
        self.enoise = noise(32)
        
    def train(self):
        
        #Train Alternating
        a = self.train_dis()
        b = self.train_gen()
        c = self.train_vae()
        
        #Print info
        if self.GAN.steps % 200 == 0 and not self.silent:
            print("\n\nRound " + str(self.GAN.steps) + ":")
            print("D: " + str(a))
            print("G: " + str(b))
            print("V: " + str(c))
            s = round((time.clock() - self.lastblip) * 1000) / 1000
            print("T: " + str(s) + " sec")
            self.lastblip = time.clock()
            
            #Save Model
            if self.GAN.steps % 500 == 0:
                self.save(floor(self.GAN.steps / 10000))
                self.evaluate(floor(self.GAN.steps / 1000))
        
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
    
    def train_vae(self):
        
        train_data = [get_rand(self.ImagesA, BATCH_SIZE), noise(BATCH_SIZE)]
        
        v_loss = self.VAEModel.train_on_batch(train_data, train_data)
        
        return v_loss
    
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
        self.saveModel(self.GAN.E, "enc", num)
        

    def load(self, num): #Load JSON and Weights from /Models/
        steps1 = self.GAN.steps
        
        self.GAN = None
        self.GAN = GAN()

        #Load Models
        self.GAN.G = self.loadModel("gen", num)
        self.GAN.D = self.loadModel("dis", num)
        self.GAN.E = self.loadModel("enc", num)
        
        self.GAN.steps = steps1
        
        self.generator = self.GAN.generator()
        self.DisModel = self.GAN.DisModel()
        self.AdModel = self.GAN.AdModel()
        self.VAEModel = self.GAN.VAEModel()
    
        
    def sample(self, n):
        
        return self.generator.predict(noise(n))
    
    def instance_noise(self):
        
        self.AmagesA = np.array(self.AmagesA)
        
        self.ImagesA = self.AmagesA + np.random.uniform(-self.noise_level, self.noise_level, size = self.AmagesA.shape)
        
        
        
        
if __name__ == "__main__":
    model = WGAN(lr = 0.0001, silent = False)
    
    while(model.GAN.steps < 5000000):
        model.train()


