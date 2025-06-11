from openai import OpenAI
import base64
from PIL import Image, ImageDraw
from io import BytesIO
import os

# it is the detailed and employed instruction, it is the basis of our method
text =  '''
Task Instruction: Door Localization and Classification

You are given a list of images:

- Image 1: A complete architectural floor plan.
- Images 2 to last: Cropped images each showing a specific door extracted from Image 1.

---

Door Categories

1. Main Entry Door
2. Bedroom Door
3. Bathroom or Washroom Door
4. Kitchen Door
5. Living Room or Dining Room Door
6. Laundry or Utility Room Door
7. Garage Door
8. Balcony or Terrace Door
9. Emergency Exit Door

---

Your Tasks

1. Locate each cropped image in Image 1 (Image 2 to last):

2. Identify the door type:
   - If multiple doors are visible in the cropped image, focus only on the center door with a red rectangle.
   - First, choose one category from (1) to (6), (7), or (8).
   - Then determine if the door qualifies as an Emergency Exit Door (9).
   - Combine the two results in your final classification (e.g., (1^9)).

---

Definitions

Main Entrance Door  
A door should be classified as a main entrance door if it meets all the following:  
- Positioned on the outer boundary of the building layout.  
- The primary access point into the building from outside.  
- Often connected directly to an entry hall, foyer, or corridor.  
- Larger or more prominent than secondary doors.  
- Not a back door, emergency exit, or internal door between rooms.

Emergency Exit Door  
A door should be classified as an emergency exit door if it meets all the following:  
- Connects an interior space directly to the outside.  
- Leads outdoors without passage through any other enclosed space.  
- May display emergency signage or symbols (e.g., “EXIT” signs, arrows).  
- Can include the main entry door if it serves as an evacuation path.  
- Often located at corridor ends, near stairwells, or service areas designed for emergency evacuation.

Other Door Types  
- Bedroom Door (2): Leads to private sleeping areas, often adjacent to closets or bathrooms.  
- Bathroom or Washroom Door (3): Provides access to rooms with toilets, sinks, showers, or bathtubs.  
- Kitchen Door (4): Opens to cooking or food prep areas, identifiable by counters or appliances.  
- Living Room or Dining Room Door (5): Leads to communal gathering areas, often spacious or open-plan.  
- Laundry or Utility Room Door (6): Opens to service areas with washing machines, dryers, or utility sinks.  
- Garage Door (7): Connects to indoor vehicle storage areas, typically near exterior driveways.  
- Balcony or Terrace Door (8): Provides access to external elevated platforms or terraces, usually outdoors.

---

✅ Output Format
At the end, utilize a single line to output the predicted category number(s) for each image as follows:
(1)Different categories for the same door are connected using ^.
(2)Different images are separated by commas.
(3)The entire output is wrapped in double parentheses (( )).

📌 Examples:
1.A bedroom door: ((2))
2.A main entry door that is also an emergency exit: ((1^9))
3.Multiple images:
Image 2: Bedroom door → 2
Image 3: Balcony door → 8
Image 4: Garage door and emergency exit → 7^9
Final output: ((2,8,7^9))
'''


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


client = OpenAI(api_key="")

# f = open('test.txt', 'r')
# lines = f.readlines()
# f.close()
#
# image_list = dict()
#
# for line in lines:
#     name = line.strip().split('/')
#     name = name[1] + '_' + name[2] + '.png'
#     image_list[name] = 0

f = open('../Co-DETR/result.txt', 'r')
lines = f.readlines()
f.close()

for idx, line in enumerate(lines):
    content = list()
    text_dict = dict()
    text_dict['type'] = 'text'
    text_dict['text'] = text
    content.append(text_dict)
    line_split = line.split(";")
    img = line_split[0]
    # try:
    #     temp = image_list[img.split('/')[-1]]
    # except KeyError:
    #     continue
    f = open('gpt-4.1_result.txt', 'a')
    f.write(img + "'''")
    image1_base64 = encode_image(img)
    image1_data_url = f"data:image/jpeg;base64,{image1_base64}"
    image1_dict = dict()
    image1_dict['type'] = "image_url"
    image1_dict['image_url'] = dict()
    image1_dict['image_url']['url'] = image1_data_url
    content.append(image1_dict)#it is the full image (floor plan)
    image = Image.open(img)
    image.save(img.split('/')[-1])
    width, height = image.size
    for i in range(1, len(line_split)-1):#the first one is image name and the last one has no value
        pos = line_split[i]
        pos_split = pos.split(',')
        score = pos_split[4]
        if float(score) < 0.3:#remove bounding boxes whose scores are less than 0.3
            continue
        x1, y1, x2, y2 = pos_split[0:4]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        x_min, y_min, x_max, y_max = x1, y1, x2, y2
        x1, y1, x2, y2 = max(0, x1 - 200), max(0, y1 - 200), min(width, x2 + 200), min(height, y2 + 200)#enlarge the bounding box with 200 pixels at every side
        image_copy = image.copy()
        draw = ImageDraw.Draw(image_copy)
        rectangle = (x_min, y_min, x_max, y_max)
        red = (255, 0, 0)
        draw.rectangle(rectangle, outline=red, width=1)#draw box in red
        image_crop = image_copy.crop((x1, y1, x2, y2))#then crop
        image_crop.save(img.split('/')[-1].split('.')[0] + '_' + str(i) + '.png')
        buffered = BytesIO()
        image_crop.save(buffered, format="PNG")
        image_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        image2_data_url = f"data:image/jpeg;base64,{image_str}"
        image2_dict = dict()
        image2_dict['type'] = "image_url"
        image2_dict['image_url'] = dict()
        image2_dict['image_url']['url'] = image2_data_url
        content.append(image2_dict)#it is the cropped image with the focused box in red color

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "user",
                "content": content
            }
        ]
    )# it is invoking GPT-4.1 API
    f.write(response.choices[0].message.content)# it is the result
    print(idx)
    f.write("'''")
    f.write('\n')
    f.close()
print('.....................end............................')

