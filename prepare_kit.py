def cut_bbox(paths, path_for_dir, labels_path, output_directory, new_size_x, new_size_y):
    pic_path = os.path.join(output_directory, "crop_pic")
    label_path = os.path.join(output_directory, "crop_label")
    current_image_number = 0
    with open(labels_path) as f:
        data = json.load(f)
        for path in paths:
            image_name = path
            if ('rgb.png' in image_name):
                path = os.path.join(path_for_dir, path)
                with Image.open(path) as img:
                    src_w, src_h = img.size
                    tar = None
                    for i in data['images']:
                        if i['file_name'] == image_name:
                            tar = i['id']
                            break
                    for i in data['annotations']:
                        if i['iscrowd'] == 0:
                            if i['image_id'] == tar:
                                box_x, box_y, box_w, box_h = i['bbox']
                                current_image_number += 1
                                crop_box = (box_x, box_y, box_x+box_w, box_y+box_h)
                                cropped_img = img.crop(crop_box)

                                
                                size = max(box_w+10, box_h+10)

                               
                                background = Image.new('RGB', (int(size), int(size)), (0, 0, 0))

                                
                                paste_x = int((size-box_w)//2)
                                paste_y = int((size-box_h)//2)

                               
                                background.paste(cropped_img, (paste_x, paste_y))

                                
                                new_image = background.resize((new_size_x, new_size_y), Image.LANCZOS)

                                # Scale the joint labels to match the new size
                                l_of_key_points = i['keypoints']
                                l_of_confidence = []
                                x_y = []
                                for j in range(0, len(l_of_key_points), 3):
                                    confidence = l_of_key_points[j+2]
                                    x = l_of_key_points[j]
                                    y = l_of_key_points[j+1]
                                    #if confidence > 0:
                                        
                                    x_scaled = ((x - box_x + paste_x) * new_size_x) / size
                                    y_scaled = ((y - box_y + paste_y) * new_size_y) / size
                                    x_y.append(round(x_scaled))
                                    x_y.append(round(y_scaled))
                                    l_of_confidence.append(confidence)
                                    #else:
                                     #   x_y.append(-1)
                                      #  x_y.append(-1)
                                       # l_of_confidence.append(confidence)

                               
                                new_dictionary = {"x_y": x_y, "confidence": l_of_confidence}

                               
                                json_data = json.dumps(new_dictionary)

                               
                                new_image.save(f"{pic_path + os.path.sep + image_name[:len(image_name)-4]}_{current_image_number}.png")

                                # write the JSON data to a new file
                                with open(f"{label_path + os.path.sep + image_name[:len(image_name)-4]}_{current_image_number}.json", "w") as file:
                                    file.write(json_data)