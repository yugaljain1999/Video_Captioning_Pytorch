# Video_Captioning_Pytorch
Video captioning on MSR-VTT Dataset

### Dataset Download
* Training and validation videos - https://drive.google.com/drive/folders/1JRCuI0_osjj9lFieajyIujSP0fYYvaqa?usp=sharing
* Captions json - https://www.dropbox.com/s/tzmd59g4tj1x71m/captions.json?dl=0

### Prepare Dataset - 
* Download videos and captions json from above link in colab using !wget command as dataset size is ~4GB.
* Make a folder 'TrainValVideo' and extract those videos in this folder.
* Run python pre_process_videos.py to extract features of each video, it will save all features in data/feats/inception_v3 folder.
* Run python train.py for training encoder-decoder rnn based model on MSR-VTT Dataset. For now inception_v3 pretrained model is used to extract features from videos, to use another pretrained model just change model name in line 64 in pre_process_video.py.


For any query feel free to raise an issue.
