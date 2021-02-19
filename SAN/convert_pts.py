import os, json
root_dir = "/data/keypoints/keypoints_annotations_json/"
annotations_output_dir = "/data/keypoints/keypoints_annotations"
version = "version: 1\n"
n_points = "n_points: 70\n"
for (root, dirs, files) in os.walk(root_dir):
    for file in files:
        if file.split('.')[-1] == 'json':
            with open(root+"/"+file, 'r') as f:
                json_data = json.load(f)
            points = json_data["ObjectInfo"]["KeyPoints"]["Points"]
            points_dict = zip(points[::2], points[1::2])
            df = open(annotations_output_dir+"/"+file[:-4]+"pts", 'w')
            df.write(version)
            df.write(n_points)
            df.write("{\n")
            for x, y in points_dict:
                x = float(x)
                y = float(y)
                data = "%.3f %.3f\n" % (x, y)
                df.write(data)
            df.write("}")
            df.close()
