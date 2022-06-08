# Detection and Tracking with Loomo
**Nicolaj Schmid,Danial Zendehdel, Ekrem Yüksel, Thomas Jaouën**

Except that Loomos can provide mobility service, and among all of its features, it allows the user to deploy its model and test it on a real-life gadget.   
This project aims to detect a person as the person of interest, which has been done here by the [Openpifpaf](https://openpifpaf.github.io/intro.html) algorithm and then Track the recognized person by the [yolov4-deepsort](https://github.com/theAIGuysCode/yolov4-deepsort) algorithm.
## Detect the person of interset 
Openpifpaf is used In order to recognise a person as a target for the tracking part. This is done by introducing a pose rule in which the `left wrist is higher than left shoulder and the right wrist is lower than right shoulder`.
As the loomos work with low resolution the images are downscaled to 160*120. 

![test_img_pifpaf](https://user-images.githubusercontent.com/49899830/172439005-eac9b6cf-7391-4709-bf51-88fb76aa50da.jpg)

## Tracking 
The Tracking is based on the Yolovv4-Deepsort, robust in low-resolution images and Tested on `V100`. 


https://user-images.githubusercontent.com/49899830/172440576-a3a108c3-909b-4b78-a738-82cc4e0a0da9.mov

# Getting Started
To get started, install the proper dependencies via Pip. <br/>
```ruby
$ virtualenv <env_name>
$ source <env_name>/bin/activate
(<env_name>)$ pip install -r path/to/requirements.txt
```
Install openpifpaf 
```ruby
pip3 install openpifpaf
```
# Running the openpifpaf
For images 
```ruby
python openpifpaf_api.py --datapath data/test_img2.png
```
# Running the Pipline on Loomo
```ruby
 python client.py --ip-address 128.179.159.152 --d 4
 ```


           


