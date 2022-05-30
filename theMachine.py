import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg

from core.utils import get_anchors

from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

#TODO: need to add reinitialization function


def findPOILabel(poi_bbox, detection_list, iou_metric):
    '''
    This function determines the Person of Interest via specified metrics, x-distance or IoU metric. 
    Distance metric is implemented as following: Function takes the center coordinates of list of detected objects 
    and the Person of Interest. Then, it checks the absolute distance between each of them and 
    takes the one that has smallest distance.
    
    IoU metric is done as following, function takes the bounding box coordinates of 
    the PoI and checks the IoU with each. Then, takes the max IoU ratio and its label, and checks if it
    exceeds a certain threshold.
    '''
    
    if iou_metric == True:
        iou_th = 0.8
        max_iou = 0
        max_iou_index = None
        for i in range(len(detection_list)):
            iou = findIOU(poi_bbox, detection_list[i])
            print('Object ID:', i, 'Intersection over Union:', iou)
            if iou >= max_iou:
                max_iou = iou
                max_iou_index = i

        if max_iou >= iou_th: 
            print('Person of Interest is found father. ID:', max_iou_index + 1)
            return max_iou_index + 1, True
        else: 
            print('Person of Interest is not found father. Samaritan is hiding the PoI.')
            return None, False
    else:
        dist = []
        print("POI CENTER: [", poi_bbox[0], ", ", poi_bbox[1], "]")
        print("OBJECT COORDS", detection_list)
        for i in range(len(detection_list)):
            dist.append(np.abs(detection_list[i][0] - poi_bbox[0]))
            print("CENTER COOOORDS", detection_list[i][0])
            print("DIST VALUES", dist)
        return dist.index(min(dist)) + 1, True

    
def findIOU(poi_bb, obj_bb):
    '''
    This function computes the intersection over union
    Input: parameters are person of interest and detected object bounding boxes as bbox[x_center, y_center, width, height]
    Output: iou (intersection over union) ratio in range [0, 1]
    Coordinates of bounding the frame is defined as from top left corner of the frame towards down and right.
    First, intersection area is found.    
    (bbox[0] - bbox[3]/2, bbox[1] - bbox[4]/2) is the top left corner (x,y) coordinates of bounding box.
    (bbox[0] + bbox[3]/2, bbox[1] + bbox[4]/2) is the bottom right corner (x,y) coordinates of bounding box.
    Intersection box coordinates are:
    inter_x_topleft = max(poi_topleft_x, obj_topleft_x),    inter_y_topleft = max(poi_topleft_x, obj_topleft_x)
    inter_x_botright = min(poi_botright_x, obj_botright_x), inter_y_botright = min(poi_botright_x, obj_botright_x)
    inter_width = abs(inter_x_topleft - inter_x_botright)
    inter_height = abs(inter_y_topleft - inter_y_botright)
    intersection = inter_width * inter_height
     '''
    intersection = \
        max(0, min(poi_bb[0] + poi_bb[2]/2, obj_bb[0] + obj_bb[2]) - max(poi_bb[0] - poi_bb[2]/2, obj_bb[0])) * \
        max(0, min(poi_bb[1] + poi_bb[3]/2, obj_bb[1] + obj_bb[3]) - max(poi_bb[1] - poi_bb[3]/2, obj_bb[1]))
    # computing union area
    union = poi_bb[2] * poi_bb[3] + obj_bb[2] * obj_bb[3] - intersection

    return intersection/union

class theMachine:
    def __init__(self):
        self.max_cos_dist = 0.4
        self.nn_budget = None
        self.nms_max_overlap = 1.0
        self.score_th = 0.5
        self.iou_th = 0.45

        #loading standard tensorflow model
        yolo_weight_path = './checkpoints/yolov4-tiny-416'
        self.yolo = tf.saved_model.load(yolo_weight_path, tags=[tag_constants.SERVING])
        self.infer = self.yolo.signatures['serving_default']

        deepsort_model_name ='model_data/mars-small128.pb'        
        self.encoder = gdet.create_box_encoder(deepsort_model_name, batch_size=1)
        # calculate cosine distance metric
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", self.max_cos_dist, self.nn_budget)
        # initialize tracker
        self.tracker = Tracker(metric)

        self.STRIDES = np.array(cfg.YOLO.STRIDES_TINY)
        self.ANCHORS = get_anchors(cfg.YOLO.ANCHORS_TINY, True)
        self.XYSCALE = cfg.YOLO.XYSCALE_TINY 

        self.input_width = 416
        self.input_height = 416

        #POI ID variables
        self.poi_id = None
        self.poi_lost_count = 0
        self.max_frames_lost = 20
        self.poi_bbox = None
        self.tracking_poi = False

        self.dont_show = False
        self.info = True
        
    def set_input_size(self, width, height):
        self.input_width = width
        self.input_height = height

        

    # sets the POI ID
    def set_poi_id(self, poi_id):
        self.poi_id = poi_id

    def run_detections(self, image_data):
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]
        
    def track_poi(self, img, image_data, poi_bbox):
        
        batch_data = tf.constant(image_data)
        pred_bbox = self.infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]
        # TODO: finish creating the class
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=self.iou_th,
            score_threshold=self.score_th)

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        class_names = utils.read_class_names(cfg.YOLO.CLASSES)
	# by default allow all classes in .names file
        allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
        # allowed_classes = ['person']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)
        # encode yolo detections and feed to tracker
        features = self.encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]


        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        self.bboxes = np.array([d.tlwh for d in detections])
        self.scores = np.array([d.confidence for d in detections])
        self.classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(self.bboxes, self.classes, self.nms_max_overlap, self.scores)
        self.detections = [detections[i] for i in indices]         

        # if self.poi_lost is True:
        #     print('POI IS NOT FOUND')
        #     return self.detections

        # print('\n____________________________________________________\n',
        #     self.bboxes, self.scores, self.classes)
        if self.tracking_poi is False and len(self.bboxes):
            print('\n ________________________ SEARCHING ADMIN... ________________________\n')
            self.poi_id, self.tracking_poi = findPOILabel(poi_bbox, self.bboxes, True)
            if self.tracking_poi:
                print('\n ________________________ ADMIN IS FOUND ________________________\n',self.poi_id, self.tracking_poi)
            else:
                print('\n ________________________  CANNOT FIND ADMIN ________________________\n')
                return


        self.tracker.predict()
        self.tracker.update(self.detections)

        # update tracks
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            # bbox1 = track.to_tlwh()
            # print('GELİYORU GELMEKTE OLAN', bbox)
            class_name = track.get_class()
            
            if self.tracking_poi is True and self.poi_id is track.track_id:
                print('\n ________________________ TRACKING ADMIN... ________________________\n')
                self.poi_bbox = track.to_tlwh()
            elif self.tracking_poi is False:
                # print('\n ------- poi is still unknown. tracker id',
                #  track.track_id,'poi id:', self.poi_id, ' -------- \n')                
                continue

            
            # draw bbox on screen
            if track.track_id is self.poi_id:
            # draw bbox on screen
                color = (255, 234, 0)
                color = [i * 255 for i in color]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                cv2.putText(frame, "ADMIN", (int(bbox[0]), int(bbox[1]-10)),0, 0.75, (225,225,225),2)
                return
            else:
                continue
                        # draw bbox on screen
                # color = colors[20 % len(colors)]
                # color = [i * 255 for i in color]
                # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                # cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

        # if enable info flag then print details about each track
            if self.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))


if __name__ == '__main__':

    the_machine = theMachine()

    vid = cv2.VideoCapture(0) #'./data/video/test.mp4' or '0'
    # cv2.namedWindow("Output Video", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Output Video", the_machine.input_width, the_machine.input_height) 
    
    output = True
    out = None

    # get video ready to save locally if flag is set
    # if output:
    #     # by default VideoCapture returns float instead of int
    #     width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    #     height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #     # fps = int(vid.get(cv2.CAP_PROP_FPS))
    #     # codec = cv2.VideoWriter_fourcc()
    #     out = cv2.VideoWriter(output, codec, fps, (width, height))

    frame_num = 0

    while True:
    
        return_value, frame = vid.read()
        
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed! Samaritan is interfering father.')
            break
        

        frame_size = frame
        image_data = cv2.resize(frame, (the_machine.input_width, the_machine.input_height))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()  

        poi_bbox = [160.,  36., 427., 409.]
        
        poi_bbox[0] += poi_bbox[2]/2
        poi_bbox[1] += poi_bbox[3]/2

        the_machine.track_poi(image, image_data, poi_bbox)

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if not the_machine.dont_show:
            cv2.imshow("Output Video", result)
        
        # if output flag is set, save video file
        # if True:
        #     out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break