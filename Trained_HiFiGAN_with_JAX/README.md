Based off the work here: https://arxiv.org/pdf/2010.05646
Taking inspiration of the code here: https://github.com/jik876/hifi-gan/blob/master/models.py#L4
And the code from here:

Training a model is hard. Below are the curves from the first 100 epochs on the data:

![alt text](<Firefox 2024-10-27 15.41.41.png>)

Unfortunately it seems like the Generator can't create convincing soundwaves, I noticed in the code that the loss wasn't taking into account the period discriminator.

After having corrected for this error I get these curves:

![alt text](<CleanShot 2024-10-27 at 18.09.04@2x.png>)

Still can't get it to converge... will come back to it !

What I like about JAX was that there was zero work needed to move from the M1 chip I'm using on my laptop to the L4 GPU on GCP that I'm now using to train. In the HiFiGAN paper they mention that they train the model for over 24 hours on the LJSpeech dataset of 13,000 samples. I have less samples and I'm wondering at what point this is having an impact on the training...

I'm going to download the LJDataset instead of the WolofTTS that I'm currently using for a future improvement.
