# GAN with Mnist
generating mnist by GAN, lsGAN, wGAN, wGAN_GP

### network.py
>activation functions and network function which can draw tensorflow layers by input dictionary.  now, it can draw fullyconnected, conv2D, transpose conv2D with reshape, batch nomalization.

### loss.py
>return discriminator and generator's loss of each GAN. 

### optim.py
>optimizer function which return tensorflow optimizer.

### data.py
>plot data image, generate noise, preprocess, deprocess

### train.py
>function which train network and plot outputs, save files

### main.py
>set hyperparameters
>- input (batch size, latent space dimension)
>- network (layers)
>- object function (loss i.e. set GAN type)
>- optimizer (type, learning rate)
>- training configuration (how many times to train discriminator per generator, epoch)
>- else (save directory, device type)

# output
latent dim 2, batch size 126

discriminator : conv(k:4, s:2, channel:64) - conv(k:4, s:2, channel:128) - fc(1024) - fc(1)

generator : fc(1024) - fc(4096) - transpose conv(k:4, s:2, channel:64) - transpose conv(k:4, s:2, channel:1)

![Alt text](latent_dim_2_batch_126_gan.JPG)
