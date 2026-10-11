from PIL import Image
import os
import json


def resize_image(image, dic_labels, new_size_x, new_size_y):

    new_size = (new_size_x, new_size_y)
    size_src = image.size
    resized_image = image.resize(new_size)

    # Scale the joint labels to match the new size
    scale_x = resized_image.size[0] / size_src[0]
    scale_y = resized_image.size[1] / size_src[1]

    for joint in range(len(dic_labels['x_y'])):
        if (joint % 2 == 0):
            dic_labels['x_y'][joint] = int(dic_labels['x_y'][joint]*scale_x)
        elif (joint % 2 == 1):
            dic_labels['x_y'][joint] = int(dic_labels['x_y'][joint]*scale_y)

    # return the resized image to a new file
    return resized_image, dic_labels
def cut_bbox(paths, path_for_dir, labels_path, output_directory, new_size_x, new_size_y):
    pic_path = os.path.join(output_directory, "crop_pic")
    label_path = os.path.join(output_directory, "crop_label")
    # os.mkdir(pic_path)
    # os.mkdir(label_path)
    current_image_number = 0
    with open(labels_path) as f:
        data = json.load(f)
        for path in paths:
            image_name = path
            if ('rgb.png' in image_name):
                path = os.path.join(path_for_dir, path)
                with Image.open(path) as img:
                    src=img.size
                    tar = None
                    for i in data['images']:
                        # print(tar)
                        if i['file_name'] == image_name:

                            tar = i['id']
                            break
                    for i in data['annotations']:
                        if i['iscrowd'] == 0:

                            if i['image_id'] == tar:

                                box = i['bbox']
                                current_image_number += 1

                                # crop the image based on the bounding box and add 5 pixels width and height to make it a square
                                x, y, w, h = box
                                crop_box = (x-5, y-5, x+w+5, y+h+5)
                                cropped_img = img.crop(crop_box)

                                # calculate the size of the square image to be filled with black
                                size = max(w+10, h+10)

                                # create a new image with the calculated size filled with black
                                background = Image.new('RGB', (int(size), int(size)), (0, 0, 0))

                                # calculate the position to paste the cropped image onto the black background image
                                paste_x = int((size-w)//2)
                                paste_y = int((size-h)//2)

                                # paste the cropped image onto the black background image
                                background.paste(cropped_img, (paste_x, paste_y))

                                # resize the image to the new size
                                area = (box[0]-5, box[1]-5, box[0]+5 + w, box[1]+5 + h)
                                x_y=[]
                                confidence=[]
                                # Scale the joint labels to match the new size
                                i['bbox'][0] = 0
                                i['bbox'][1] = 0
                                l_of_key_points = i['keypoints']
                                for j in range(len(l_of_key_points)):
                                    if (j+1) % 3 == 1:
                                        l_of_key_points[j] = max( 0, l_of_key_points[j]-area[0])
                                        x_y.append(l_of_key_points[j])
                                    elif (j+1) % 3 == 2:
                                        l_of_key_points[j] = max( 0, l_of_key_points[j]-area[1])
                                        x_y.append(l_of_key_points[j])
                                    else:
                                        confidence.append(l_of_key_points[j])
                                new_dictionary = {"x_y": x_y,"confidence":confidence}
                                resized_image, new_dictionary = resize_image(background, new_dictionary, new_size_x, new_size_y)

                                

                                new_bbox = [paste_x, paste_y, paste_x + w, paste_y + h]

                                # Scale the bounding box coordinates to match the resized image
                                scale_x = resized_image.size[0] / background.size[0]
                                scale_y = resized_image.size[1] / background.size[1]
                                new_bbox[0] = int(new_bbox[0] * scale_x)
                                new_bbox[1] = int(new_bbox[1] * scale_y)
                                new_bbox[2] = int(new_bbox[2] * scale_x)
                                new_bbox[3] = int(new_bbox[3] * scale_y)

                                # Update the annotation dictionary with the new bounding box coordinates
                                i['bbox'] = new_bbox
                                l_of_key_points = new_dictionary['x_y']
                                for j in range(len(l_of_key_points)):
                                    if (j+1) % 2 == 1:
                                        l_of_key_points[j] = int(l_of_key_points[j] * scale_x)
                                    elif (j+1) % 2 == 0:
                                        l_of_key_points[j] = int(l_of_key_points[j] * scale_y)

                                # Convert the dictionary to a JSON string
                                json_data = json.dumps(new_dictionary)

                                # save the resized image to a new file
                                resized_image.save(f"{pic_path + os.path.sep + image_name[:len(image_name)-4]}_{current_image_number}.png")

                                # write the JSON data to a new file
                                with open(f"{label_path + os.path.sep + image_name[:len(image_name)-4]}_{current_image_number}.json", "w") as file:
                                    file.write(json_data)

    