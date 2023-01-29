!git clone https://github.com/ultralytics/yolov5  # clone repo
!pip install -r yolov5/requirements.txt  # install dependencies
%cd yolov5

import torch
from IPython.display import Image, clear_output  # to display images
from utils.google_utils import gdrive_download  # to download models/datasets

clear_output()
print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))

!python detect.py --weights yolov5s.pt --img 416 --conf 0.4 --source ./inference/images/
Image(filename='inference/output/zidane.jpg', width=600)

# Example syntax (do not run cell)
!python detect.py --source ./file.jpg  # image 
                           ./file.mp4  # video
                           ./dir  # directory
                           0  # webcam
                           'rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa' # rtsp
                           'http://112.50.243.8/PLTV/88888888/224/3221225900/1.m3u8'  # http
                           
# Download COCO val2017
gdrive_download('1Y6Kou6kEB0ZEMCCpJSKStCor4KAReE43','coco2017val.zip')  # val2017 dataset
!mv ./coco ../  # move folder alongside /yolov5

# Run YOLOv5s on COCO val2017
!python test.py --weights yolov5s.pt --data ./data/coco.yaml --img 640

# Download COCO test-dev2017
gdrive_download('1cXZR_ckHki6nddOmcysCuuJFM--T-Q6L','coco2017labels.zip')  # annotations
!f="test2017.zip" && curl http://images.cocodataset.org/zips/
f && unzip -q f && rm
f  # 7GB,  41k images
!mv ./test2017 ./coco/images && mv ./coco ../  # move images into /coco and move /coco alongside /yolov5
     

# Run YOLOv5s on COCO test-dev2017 with argument --task test
!python test.py --weights yolov5s.pt --data ./data/coco.yaml --task test


# Download tutorial dataset coco128.yaml
gdrive_download('1n_oKgR81BJtqk75b00eAjdv03qVCQn2f','coco128.zip')  # tutorial dataset
!mv ./coco128 ../  # move folder alongside /yolov5

# Start tensorboard
%load_ext tensorboard
%tensorboard --logdir runs

# Train YOLOv5s on coco128 for 5 epochs
!python train.py --img 640 --batch 16 --epochs 5 --data ./data/coco128.yaml --cfg ./models/yolov5s.yaml --weights yolov5s.pt --name tutorial --nosave --cache

Image(filename='./train_batch1.jpg', width=900)  # view augmented training mosaics

Image(filename='./test_batch0_gt.jpg', width=900)  # view test image labels

Image(filename='./test_batch0_pred.jpg', width=900)  # view test image predictions

from utils.utils import plot_results; plot_results()  # plot results.txt as results.png
Image(filename='./results.png', width=1000)  # view results.png

# Re-clone
%cd ..
!rm -rf yolov5 && git clone https://github.com/ultralytics/yolov5
%cd yolov5
     

# Apex install
git clone https://github.com/NVIDIA/apex && cd apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" . --user && cd .. && rm -rf apex
     

# Test GCP checkpoint on COCO val2017
%%shell
x=best*.pt
gsutil cp gs://*/*/weights/
x --data ./data/coco.yaml --img 736
     

# Test multiple models on COCO val2017
%%shell
for x in yolov5s yolov5m yolov5l yolov5x
do 
  python test.py --weights $x.pt --data ./data/coco.yaml --img 640 --conf 0.001
done
     
