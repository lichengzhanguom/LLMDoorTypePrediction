# The code is for predicting whether the balcony is enclosed or not as sometimes enclosed balcony is classified as emergency exit doors,
# but actually they can not access the outside
from openai import OpenAI
import base64
from PIL import Image, ImageDraw, ImageFont
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


text =  '''## 🧭 **Task: Identify the Main Entrance Door(s) in an Image**

You are given a single image containing **multiple doors**, each enclosed in a **red box**. On top of each red box is a **red number** identifying that door.

---

### 🔍 **Definition of a Main Entrance Door**

A door is considered a **main entrance door** if it meets **all** the following conditions:

- It is the main entrance of the building.
- If there are **two ET labels**, there should be **two main entrance doors**, each closest to one ET label.
- If there are **both ET and TK labels**, choose the door closest to the ET label rather than TK label.
- If there are **TK label only** (and no ET), choose the door closest to the TK label.
- If there are **no ET and no TK labels**, determine based on your knowledge.
- **Check each candidate red box to ensure it actually contains a door**. If it does not (i.e., the box encloses a space but no door), exclude it and select from other boxes if possible.
- **Check the function of the door, i.e., the room type it serves** —
  - if the door is a garage door, it should not be the main entrance door.
  - if the door is a balcony or terrace door, it should not be the main entrance door.
  - if the door is a bathroom or washroom door, it should not be the main entrance door.
- if the door serves a room that has only this door (no other doors), it should not be the main entrance door.
- if the door serves a room labeled VAR, it should not be the main entrance door.

---

### 🔍 **Selection Rule**

1. Check for ET and TK labels:

   - If there are **two ET labels**:  
     → Choose the **two red boxes** whose centers are **closest to each ET label** respectively.

   - If there is **one ET label only**:  
     → Choose the red box whose center is **closest to the ET label**.

   - If there is **TK label only** (and no ET):  
     → Choose the red box whose center is **closest to the TK label**.

   - If there are **both ET and TK labels**:  
     → Choose the red box closest to the ET label rather than the TK label.

2. If there are **no ET and no TK labels**:  
   → Choose the red box that is **most likely to be the main entrance door** based on your overall knowledge.

3. If there are **no red boxes**, or **all red boxes enclose a space but are not actual doors**, return `((0))`.

---

### 📋 **Instructions**

- Examine each numbered red box in the image.
- Verify that the box actually contains a door (not just an enclosed space).
- Verify that the door is **not a garage door**, balcony/terrace door, or bathroom/washroom door.
- Verify that the door does not serve a room that has only this door (no other doors).
- Verify that the door does not serve a room labeled VAR.
- Identify which door(s) qualify as the **main entrance door(s)** based on the criteria above.
- In **most cases**, there is **only one main entrance door**.
- In **rare cases**, there may be **no** or **two** main entrance doors.
- If none qualify, return `((0))`.
- If one qualifies, return its number as `((number))`.
- If two qualify, return both numbers as `((number1, number2))`, ordered ascending by number.

---

### ✅ **Output Format**

Return a single output in one of these formats:

((number))           # one main entrance door  
((number1, number2)) # two main entrance doors  
((0))                # no main entrance door  

*No need to provide any explanation or reason after the answer.*
'''



def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def find_outermost_pixels_by_scanning(binary):
    h, w = binary.shape
    outer_pixels = set()

    # 上
    for x in range(w):
        for y in range(h):
            if binary[y, x] == 0:
                outer_pixels.add((x, y))
                break
    # 下
    for x in range(w):
        for y in range(h - 1, -1, -1):
            if binary[y, x] == 0:
                outer_pixels.add((x, y))
                break
    # 左
    for y in range(h):
        for x in range(w):
            if binary[y, x] == 0:
                outer_pixels.add((x, y))
                break
    # 右
    for y in range(h):
        for x in range(w - 1, -1, -1):
            if binary[y, x] == 0:
                outer_pixels.add((x, y))
                break

    return list(outer_pixels)


def filter_small_components(binary_img, min_area=100):
    """
    过滤连通区域面积小于 min_area 的区域，保留较大区域
    输入：二值图（0为前景）
    输出：过滤后的二值图（0为前景）
    """
    # 反转图像，方便连通域检测
    binary_inv = cv2.bitwise_not(binary_img)

    # 连接区域标记
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_inv, connectivity=8)

    filtered = np.ones_like(binary_img) * 255  # 白底

    for i in range(1, num_labels):  # 0是背景
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            filtered[labels == i] = 0  # 保留较大区域为黑色

    return filtered


client = OpenAI(api_key="")

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

threshold_value = 100  # 降低阈值

for idx, line in enumerate(lines):
    content = list()
    text_dict = dict()
    text_dict['type'] = 'text'
    text_dict['text'] = text
    content.append(text_dict)
    line_split = line.strip().strip(';').split(':')
    img = line_split[0]
    try:
        print(img)
        temp = image_list[img.split('/')[-1]]
    except KeyError:
        print(img)
        continue
    if line_split[1] == '':
        continue
    line_splits = line_split[1].split(";")
    image = cv2.imread(img)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 低阈值二值化
    _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

    # 连通区域过滤
    filtered = filter_small_components(binary, min_area=500)

    # 反转，线条为黑，背景为白
    binary_clean = filtered

    outer_pixels = find_outermost_pixels_by_scanning(binary_clean)

    canvas = np.ones_like(image) * 255
    for x, y in outer_pixels:
        cv2.circle(canvas, (x, y), 5, (0, 255, 0), -1)

    red_boxes = line_splits
    for i, box in enumerate(red_boxes, 1):
        x1, y1, x2, y2, _ = box.split(',')
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        roi = canvas[y1:y2, x1:x2]

        # 查找是否包含纯蓝色像素
        blue_pixels = np.where(
            (roi[:, :, 0] == 0) & (roi[:, :, 1] == 255) & (roi[:, :, 2] == 0)
        )

        if len(blue_pixels[0]) > 0:
            color = (0, 0, 255)  # 红框（BGR）

            # 绘制框
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            # 绘制文本
            cv2.putText(
                image,
                str(i),
                (x1 + 5, y1 - 10),  # 文字位置
                cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                0.8,  # 字体大小
                (0, 0, 255),  # 字体颜色（绿色）
                2  # 字体厚度
            )

    cv_img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cv_img_rgb)
    image = pil_img
    width, height = image.size
    f = open('gpt-4.1_result.txt', 'a')
    f.write(img + "'''")

    image.save('output/' + img.split('/')[-1])
    buffered = BytesIO()
    image.save(buffered, format="PNG")
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
    print(idx)
print('.....................end............................')
