####################### Preprocessing frames of each video ####################
import os
import glob
import torch
import torchvision.models as models
from torch import nn
import numpy as np
import shutil
import subprocess
import pretrainedmodels 
from pretrainedmodels import utils
C,H,W = 3,299,299
def extract_frames(vid,dir):
    with open('null', "w") as ffmpeg_log:
        if os.path.exists(dir):
            print(" cleanup: " + dir + "/")
            shutil.rmtree(dir)
        os.makedirs(dir)
        video_to_frames_command = ["ffmpeg",
                                    # (optional) overwrite output file if it exists
                                    '-y',
                                    '-i', vid,  # input file
                                    '-vf', "scale=400:300",  # input file
                                    '-qscale:v', "2",  # quality for JPEG
                                    '{0}/%06d.jpg'.format(dir)]
        subprocess.call(video_to_frames_command,
                        stdout=ffmpeg_log, stderr=ffmpeg_log)

def extract_features(model,load_image_func,video_dir,output_dir,model_name='inception_v3',frame_steps=40):
    model.eval()
    print(output_dir)
    #if not os.path.isdir(output_dir):
    #    os.makedirs(output_dir)
    videos = glob.glob(os.path.join(video_dir,'*.mp4')) # list of all video names
    #print(videos)
    for id,video in enumerate(videos):
        print(video)
        video_id =  video.split('/')[-1].split('.')[0] # just name of video not extension
        video_id = video_id.split('/')[-1]
        video_id = video_id.split('/')[-1]
        print(video_id)
        dst = model_name + '_' + video_id # e.g inception_v3_4545454
        # extract frames 
        extract_frames(video,dst)
        images_list = glob.glob(os.path.join(dst,'*.jpg')) # for each video_id there are frames saved at outpur directories(dst)
        # do sampling for images
        samples = np.round(np.linspace(0,len(images_list)-1,frame_steps)) # for each number of frame steps,images are sampled
        images_list = [images_list[int(sample)] for sample in samples]
        images = torch.zeros((len(images_list),C,H,W))
        for idx_image in range(len(images_list)):
            img = load_image_func(images_list[idx_image])
            images[idx_image] = img # now features of each image for each video is saved to temporary images zero tensors
        # now find features of each image using pretrained model
        with torch.no_grad():
            img_fts = model(images).squeeze().cuda()
        img_fts = img_fts.cpu().numpy()
        out_file = os.path.join(output_dir,'video{}'.format(id)+'.npy') # here features of frames is saved in .npy format
        #print(img_fts)
        np.save(out_file,img_fts)

        shutil.rmtree(dst) # here dst is directory where frames of each video are saved individually


model = pretrainedmodels.inceptionv3(pretrained='imagenet') # load pretrained inception_v3 model
#print(model)
load_image_func = utils.LoadTransformImage(model) # for inception_v3 model load transformed images
model.last_linear = utils.Identity()
model = nn.DataParallel(model)
model.cuda()
###### Calling extract_features function ################
extract_features(model,load_image_func,'TrainValVideo/TrainValVideo','data/feats/inception_v3')