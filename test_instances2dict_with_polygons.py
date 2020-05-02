from instances2dict_with_polygons import instances2dict_with_polygons
import os
import json

if __name__ == "__main__":
    path = '/home/kento/davis_ws/cityscapes/gtFine/train/aachen/'
    fileList = os.listdir(path)
    fileList = [(path+x) for x in fileList]
    dict_city = instances2dict_with_polygons(fileList[:1], True)
    json_city = json.dumps(dict_city)
    print(dict_city)
    with open('json_city.json', 'w') as f:
        json.dump(json_city, f)