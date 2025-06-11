from openai import OpenAI
import base64
from PIL import Image, ImageDraw
from io import BytesIO
import os
import cv2
import numpy as np

def remove_white_margin(img, threshold=240):
    """
    Remove white margins from an image by detecting the bounding box of non-white pixels.

    Parameters:
    - img: np.ndarray (H, W, C)
    - threshold: int (0–255), brightness above which pixels are considered white

    Returns:
    - cropped_img: np.ndarray
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create mask of non-white pixels
    mask = gray < threshold

    # Find coordinates of non-white pixels
    coords = np.argwhere(mask)

    if coords.size == 0:
        return img  # image is all white

    # Get bounding box of content
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Crop the image
    cropped = img[y_min:y_max+1, x_min:x_max+1]
    return cropped, y_min, x_min, y_max, x_max

text =  '''
🧭 Task: Judge Emergency Exit Door

You are given an image of a complete architectural floor plan.
A red box marks a specific door.

🔍 Your Task:
Determine whether the door in the red box allows direct escape to the outside (i.e., whether it qualifies as an emergency exit).

✅ Output Format:
Respond strictly with:
((Yes)) – if the door can be used as an emergency exit  
((No)) – if the door cannot be used as an emergency exit  

Do not include any other text.

📌 Classification Rules:

1. Always classify as ((Yes)) if the door is:
   - A **main entrance door**
   - A **garage door**

   ↪ These are always emergency exits. Do NOT analyze further.

2. Classify as ((Yes)) if the door is:
   - A **balcony or terrace door**, 
     unless it is clearly blocked or enclosed (e.g., by walls or glass).

3. Classify as ((No)) if the door:
   - Leads to another interior room or hallway
   - Does NOT allow direct escape outdoors

🔔 Notes:
- Be strict and consistent.
- Only consider the door marked in the red box.
- If it is a main entry or garage door, classify it immediately as ((Yes)).
'''


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


client = OpenAI(api_key="")

f = open('train.txt', 'r')
lines = f.readlines()
f.close()

image_list = dict()

for line in lines:
    name = line.strip().split('/')
    name = name[1] + '_' + name[2] + '.png'
    image_list[name] = 0

# An example is /home/lichezhang/cubicasa5k1/high_quality_architectural_8269.png:169,448,269,557,0.8100974,3;334,454,442,569,0.31040218,1^9
# 169,448,269,557,0.8100974,3 is x1, y1, x2, y2, score, category respectively
f = open('filter.txt', 'r')#This is the result after first LLM door type prediction
lines = f.readlines()
f.close()

for idx, line in enumerate(lines):
    line_split = line.strip().strip(';').split(':')
    img = line_split[0]
    line_splits = line_split[1].split(";")
    try:
        temp = image_list[img.split('/')[-1]]
    except KeyError:
        continue
    image = cv2.imread(img)
    cropped, y_min, x_min, y_max, x_max = remove_white_margin(image)#remove white margin as sometimes the floor plan only accounts for a small part of the whole floor plan, which makes doors very small.
    cv_img_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cv_img_rgb)
    image = pil_img
    width, height = image.size
    for i in range(len(line_splits)):
        pos = line_splits[i]
        pos_split = pos.split(',')
        score = pos_split[4]
        if float(score) < 0.3:
            continue
        cate = pos_split[5]
        print(cate)
        if '9' not in cate:
            if len(cate.split('^')) > 1:
                print(cate)
            else:
                if cate == '1' or cate == '7' or cate == '8':#only double check main entrance door, garage door and balcony or terrace door
                    f = open('gpt-4.1_result_refine.txt', 'a')
                    f.write(img + "'''")
                    if cate == '1':
                        text = text + 'Now I tell you the door type is the main entrance door.'
                    elif cate == '7':
                        text = text + 'Now I tell you the door type is the garage door.'
                    elif cate == '8':
                        text = text + 'Now I tell you the door type is the balcony or terrace door.'
                    content = list()
                    text_dict = dict()
                    text_dict['type'] = 'text'
                    text_dict['text'] = text
                    content.append(text_dict)
                    f.write("^^" + str(i) + "^^")
                    x1, y1, x2, y2 = pos_split[0:4]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    x1, y1, x2, y2 = x1 - x_min, y1 - y_min, x2 - x_min, y2 - y_min
                    xmin, ymin, xmax, ymax = x1, y1, x2, y2
                    image_copy = image.copy()
                    draw = ImageDraw.Draw(image_copy)
                    rectangle = (xmin, ymin, xmax, ymax)
                    red = (255, 0, 0)
                    draw.rectangle(rectangle, outline=red, width=3)# draw a red box (original) on the full image and then send the full image to GPT-4.1
                    image_copy.save(img.split('/')[-1].split('.')[0] + '_' + str(i) + '.png')
                    buffered = BytesIO()
                    image_copy.save(buffered, format="PNG")
                    image_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    image2_data_url = f"data:image/jpeg;base64,{image_str}"
                    image2_dict = dict()
                    image2_dict['type'] = "image_url"
                    image2_dict['image_url'] = dict()
                    image2_dict['image_url']['url'] = image2_data_url
                    content.append(image2_dict)

                    response = client.chat.completions.create(
                        model="gpt-4.1",
                        messages=[
                            {
                                "role": "user",
                                "content": content
                            }
                        ]
                    )
                    f.write(response.choices[0].message.content)
                    f.write("'''")
                    f.write('\n')
                    f.close()
                    print(idx)
print('.....................end............................')

