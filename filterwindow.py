#The code is for filtering boxes that is detecting windows on the wall as doors. Their characteristic is obvious
import os

f = open('test.txt', 'r')
lines = f.readlines()
f.close()

image_list = dict()

for line in lines:
    name = line.strip().split('/')
    name = name[1] + '_' + name[2] + '.png'
    image_list[name] = 0

f = open('filter.txt', 'r')
lines = f.readlines()
f.close()

f = open('after_filter.txt', 'a')

for line in lines:
    name_box = line.strip().strip(";").split(';')
    name = name_box[0]
    try:
        temp = image_list[name.split('/')[-1]]
    except KeyError:
        continue
    f.write(name + ':')
    boxes = name_box[1:]
    width_height_list = list()
    for box in boxes:
        x1, y1, x2, y2, _ = box.split(',')
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        width = x2 - x1
        height = y2 - y1
        width_height_list.append(width * height)
    mean_width = sum(width_height_list) / len(width_height_list)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2, _ = box.split(',')
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        width = x2 - x1
        height = y2 - y1
        if (width * height <= mean_width and (width / height > 2 or height / width > 2)) or width / height > 3 or height / width > 3:
            print(name, i)
            continue
        f.write(box+';')
    f.write('\n')
f.close()
