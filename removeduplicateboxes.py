# The code is for removing duplicate boxes of the same door area, it can improve the accuracy by a lot
import numpy as np


def compute_iou(box1, box2):
    """
    Computes Intersection over Union (IoU) between two bounding boxes.

    Parameters:
    - box1: (x1_min, y1_min, x1_max, y1_max)
    - box2: (x2_min, y2_min, x2_max, y2_max)

    Returns:
    - iou: float in [0, 1]
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Calculate intersection coordinates
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    # Compute intersection area
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    inter_area = inter_width * inter_height

    # Compute each box's area
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    # Compute union area
    union_area = area1 + area2 - inter_area

    # Compute IoU
    if union_area == 0:
        return 0.0  # avoid division by zero
    iou = max(inter_area / area1, inter_area / area2)
    return iou


f = open('filter.txt', 'r')
lines = f.readlines()
f.close()
threshold = 20
remove_dict = dict()
for line in lines:
    name, result = line.strip().strip(';').split(':')
    result_list = result.split(';')
    remove_dict[name] = dict()
    for i, res in enumerate(result_list):
        try:
            temp = remove_dict[name][i]
            continue
        except:
            pass
        x1, y1, x2, y2, score = result_list[i].split(',')
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        for j in range(i+1, len(result_list)):
            try:
                temp = remove_dict[name][j]
                continue
            except:
                pass
            x_min, y_min, x_max, y_max, score_twice = result_list[j].split(',')
            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
            distance_x = max(0,max(x1, x_min) - min(x2, x_max))
            distance_y = max(0,max(y1, y_min) - min(y2, y_max))
            distance = np.sqrt(distance_x ** 2 + distance_y ** 2)
            if compute_iou((x1, y1, x2, y2), (x_min, y_min, x_max, y_max)) > 0.8:
                if (x_max - x_min) * (y_max - y_min) < (x2 - x1) * (y2 - y1):
                    remove_dict[name][j] = 0
                else:
                    remove_dict[name][i] = 0


for line in lines:
    name, result = line.strip().split(':')
    f = open('filter_new.txt', 'a')
    f.write(name)
    f.write(":")
    result_list = result.split(';')
    for i, res in enumerate(result_list):
        try:
            temp = remove_dict[name][i]
        except:
            f.write(result_list[i])
            f.write(";")
    f.write("\n")
    f.close()
