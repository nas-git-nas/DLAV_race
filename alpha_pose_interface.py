import argparse
import torch

from alphapose.utils.config import update_config
from alpha_pose import SingleImageAlphaPose

class AlphaPoseInterface():
    def __init__(self):
        # init. alpha pose settings
        self.args, self.cfg = self.initArgParser()

        # define class for pose and face detection
        self.body_pose_detection = SingleImageAlphaPose(self.args, self.cfg)

    def initArgParser(self):
        parser = argparse.ArgumentParser(description='AlphaPose Single-Image Demo')
        parser.add_argument('--cfg', type=str, required=True,
                            help='experiment configure file name')
        parser.add_argument('--checkpoint', type=str, required=True,
                            help='checkpoint file name')
        parser.add_argument('--detector', dest='detector',
                            help='detector name', default="yolo")
        parser.add_argument('--image', dest='inputimg',
                            help='image-name', default="")
        parser.add_argument('--save_img', default=False, action='store_true',
                            help='save result as image')
        parser.add_argument('--vis', default=False, action='store_true',
                            help='visualize image')
        parser.add_argument('--showbox', default=False, action='store_true',
                            help='visualize human bbox')
        parser.add_argument('--profile', default=False, action='store_true',
                            help='add speed profiling at screen output')
        parser.add_argument('--format', type=str,
                            help='save in the format of cmu or coco or openpose, option: coco/cmu/open')
        parser.add_argument('--min_box_area', type=int, default=0,
                            help='min box area to filter out')
        parser.add_argument('--eval', dest='eval', default=False, action='store_true',
                            help='save the result json as coco format, using image index(int) instead of image name(str)')
        parser.add_argument('--gpus', type=str, dest='gpus', default="0",
                            help='choose which cuda device to use by index and input comma to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)')
        parser.add_argument('--flip', default=False, action='store_true',
                            help='enable flip testing')
        parser.add_argument('--debug', default=False, action='store_true',
                            help='print detail information')
        parser.add_argument('--vis_fast', dest='vis_fast',
                            help='use fast rendering', action='store_true', default=False)
        """----------------------------- Tracking options -----------------------------"""
        parser.add_argument('--pose_flow', dest='pose_flow',
                            help='track humans in video with PoseFlow', action='store_true', default=False)
        parser.add_argument('--pose_track', dest='pose_track',
                            help='track humans in video with reid', action='store_true', default=False)

        args = parser.parse_args(args=["--cfg", "configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml", "--checkpoint", "pretrained_models/fast_res50_256x192.pth", \
                                    "--image", "examples/demo/1.jpg", "--save_img"])
        cfg = update_config(args.cfg)

        args.gpus = [int(args.gpus[0])] if torch.cuda.device_count() >= 1 else [-1]
        #args.gpus = -1
        args.device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")
        args.tracking = args.pose_track or args.pose_flow or args.detector=='tracker'

        return args, cfg

    def bodyPoseIdentification(self, body_poses):
        vis_thres = [0.4] * self.cfg.DATA_PRESET.NUM_JOINTS

        for i, human in enumerate(body_poses["result"]):
            # verify if keypoints of shoulders and hands are detected
            if (human["kp_score"][5]<=vis_thres[5]) or (human["kp_score"][9]<=vis_thres[9]) or (human["kp_score"][6]<=vis_thres[6]) or (human["kp_score"][10]<=vis_thres[10]):
                continue
            
            # right shoulder higher than right hand and left shoulder lower than left hand
            if (human["keypoints"][5,1]>human["keypoints"][9,1]) and (human["keypoints"][6,1]<human["keypoints"][10,1]): 
                return i
        return None

    def getPOI(self, img):
        # estimate keypoints and bboxes
        body_poses = self.body_pose_detection.process(self.args.inputimg , img)

        #print(body_poses)

        # identifiy body pose
        POI_index = self.bodyPoseIdentification(body_poses)

        if POI_index == None:
            return [] # no person with pose was detected
        else:
            return body_poses["result"][POI_index]["bbox"] # bbox: [left upper corner x, left upper corner y, width, hight]



