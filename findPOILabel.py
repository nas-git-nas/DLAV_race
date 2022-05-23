import numpy as np


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
        iou_th = 0.9
        max_iou = 0
        max_iou_index = None
        for i in range(len(detection_list)):
            iou = findIOU(poi_bbox, detection_list[i])
            print('Object ID:', i, 'Intersection over Union:', iou)
            if iou >= max_iou:
                max_iou = iou
                max_iou_index = i

        if max_iou >= iou_th: 
            print('Person of Interest is found father. ID:', max_iou_index)
            return max_iou_index
        else: 
            print('Person of Interest is not found father. Samaritan is hiding the PoI.')
            return None
    else:
        dist = []
        print("POI CENTER: [", poi_bbox[0], ", ", poi_bbox[1], "]")
        print("OBJECT COORDS", detection_list)
        for i in range(len(detection_list)):
            dist.append(np.abs(detection_list[i][0] - poi_bbox[0]))
            print("CENTER COOOORDS", detection_list[i][0])
            print("DIST VALUES", dist)
        return dist.index(min(dist)) + 1

    
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
    # computing intersection area
    intersection = \
        abs(max(poi_bb[0] - poi_bb[2]/2, obj_bb[0] - obj_bb[2]/2) - min(poi_bb[0] + poi_bb[2]/2, obj_bb[0] + obj_bb[2]/2)) * \
        abs(max(poi_bb[1] - poi_bb[3]/2, obj_bb[1] - obj_bb[3]/2) - min(poi_bb[1] + poi_bb[3]/2, obj_bb[1] + obj_bb[3]/2))
    # computing union area
    union = poi_bb[2] * poi_bb[3] + obj_bb[2] * obj_bb[3] - intersection

    return intersection/union

if __name__ == '__main__':
    
    poi_bbox = [5, 5, 10, 10]
    obj_bbox = [[4.5, 4, 3, 3],[2, 4, 3, 9],[8, 8, 16, 16],[5, 5, 9, 9],[8, 3, 4, 4],[10, 10, 5, 5]]

    # print(findIOU(poi_bbox, obj_bbox))
    findPOILabel(poi_bbox, obj_bbox)