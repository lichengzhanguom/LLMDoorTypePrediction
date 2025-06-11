from openai import OpenAI
import base64
from PIL import Image, ImageDraw
from io import BytesIO
import os
import cv2
import numpy as np

text =  '''## 🧭 **Task: Count Emergency Exit Doors in an Image**

### 🔍 **Definition of an Emergency Exit Door**  
A door should be classified as an emergency exit door if it meets **all** the following:

- Connects an interior space directly to the outside.  
- Leads outdoors without passage through any other enclosed space.  
- May display emergency signage or symbols (e.g., “EXIT” signs, arrows).  
- Can include the main entry door if it serves as an evacuation path.  
- Often located at corridor ends, near stairwells, or service areas designed for emergency evacuation.

---

### 📋 **Instructions**
- Examine all doors in the image.
- Only count those that meet **all** the criteria above.
- Provide your answer in the format below.

---

## ✅ **Output Format**  
((number))
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

files = os.listdir('/home/lichezhang/cubicasa5k1')

for file in files:
    img = os.path.join('/home/lichezhang/cubicasa5k1', file)
    try:
        temp = image_list[img.split('/')[-1]]
    except KeyError:
        continue
    f = open('gpt-4.1_result.txt', 'a')
    f.write(img + "'''")
    content = list()
    text_dict = dict()
    text_dict['type'] = 'text'
    text_dict['text'] = text
    content.append(text_dict)
    image1_base64 = encode_image(img)
    image1_data_url = f"data:image/jpeg;base64,{image1_base64}"
    image1_dict = dict()
    image1_dict['type'] = "image_url"
    image1_dict['image_url'] = dict()
    image1_dict['image_url']['url'] = image1_data_url
    content.append(image1_dict)

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

