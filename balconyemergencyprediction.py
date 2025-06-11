# The code is for predicting whether the balcony is enclosed or not as sometimes enclosed balcony is classified as emergency exit doors,
# but actually they can not access the outside
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


text =  '''### Task: Tell me whether the balcony or terrace near the red-marked door is **enclosed**.

- **Enclosed** means the balcony or terrace is fully surrounded by walls or barriers, restricting direct outdoor access.

---

### Output Format  
`((yes))` if the balcony or terrace is enclosed  
`((no))` if it is not enclosed (i.e., accessible to the outside)
'''


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


client = OpenAI(api_key="")

f = open('test.txt', 'r')
lines = f.readlines()
f.close()

image_list = dict()

for line in lines:
    name = line.strip().split('/')
    name = name[1] + '_' + name[2] + '.png'
    image_list[name] = 0

f = open('after_filter.txt', 'r')
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
    cropped, y_min, x_min, y_max, x_max = remove_white_margin(image)
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
        if '9' in cate and '8' in cate:
            content = list()
            text_dict = dict()
            text_dict['type'] = 'text'
            text_dict['text'] = text
            content.append(text_dict)
            f = open('gpt-4.1_result.txt', 'a')
            f.write(img + "'''")
            f.write("^^" + str(i) + "^^")
            x1, y1, x2, y2 = pos_split[0:4]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            xmin, ymin, xmax, ymax = x1, y1, x2, y2
            image_copy = image.copy()
            draw = ImageDraw.Draw(image_copy)
            rectangle = (xmin, ymin, xmax, ymax)
            red = (255, 0, 0)
            draw.rectangle(rectangle, outline=red, width=3)
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
            f.write("\n")
            f.close()
            print(idx, i)
print('.....................end............................')

