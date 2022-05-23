from alpha_pose_interface import AlphaPoseInterface
import cv2


# TODO: general
# find better solution than using global variables

# create alphapose and tracker class instances
alpha_pose = AlphaPoseInterface()

# TODO: Tracking
#tracking = Tracker()

POI_detected = False




def pipline(img):
    if not POI_detected:
        POI_bbox = alpha_pose.getPOI(img)

        # TODO: Tracking
        #bbox_center = tracking.trackerIdentify(img, POI_bbox)

    else:
        pass
        # TODO: Tracking
        #bbox_center = tracking.tracker(img)

    # for testing return results from alpha pose
    bbox_center = POI_bbox

    return bbox_center

def main():
    # load testing img
    img = cv2.imread("test_img.png")

    # run through pipline (part that will run on Loomo)
    bbox = pipline(img)
    
    # if bbox is not empty list, then draw it and save img
    if bbox:
        print(f"POI was detected: {bbox}")

        # draw bbox
        start_point = ( int(bbox[0]), int(bbox[1]) )
        end_point = ( int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3]) )
        img_bbox = cv2.rectangle(img, start_point, end_point, color=(0, 255, 0), thickness=2)

        cv2.imwrite("test_img_bbox.jpg", img_bbox)
    else:
        print(f"POI was NOT detected: {bbox}")

if __name__ == "__main__":
    main()
