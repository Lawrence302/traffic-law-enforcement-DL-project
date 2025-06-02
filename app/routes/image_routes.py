from flask import Flask , render_template , url_for, redirect , send_from_directory, jsonify, Blueprint, request
import os
import cv2
import torch
import numpy as np
import pandas as pd
import math
import sqlite3
import uuid
import pathlib

from  app import my_model

# Fix for PosixPath error on Windows
pathlib.PosixPath = pathlib.WindowsPath

# database initialization function

image_routes = Blueprint('image_routes', __name__)

# Paths to the upload directories
IMAGE_FOLDER = 'app/uploads/images'
image_name = None
violation_folder = 'app/output/violations'

# function to calculate distance
   
def distance(x1,y1,x2,y2):
    sum = ((x1-x2)**2) + ((y1-y2)**2)
    dist = math.sqrt(sum)
    return dist


# Route to handle image processing
@image_routes.route('/processing_page/<filename>')
def process_image_page(filename):
    global image_name
    image_name = filename
    print(filename,' received here ')
    
    
    return render_template('file-upload-detection.html', filename=filename, is_video=False , is_image=True )
   


@image_routes.route('/process_image/<filename>')
def process_image(filename):
   
    # try:
        filename = filename
        print("image path is : ", filename)
        
        img = cv2.imread(os.path.join('app/static/uploads/images', filename))
        # preparing image for processing
        img2 = img.copy()
        img2 = cv2.resize(img2, (640, 640))
        img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))
        
        

        # loading the model
        # model = torch.hub.load("ultralytics/yolov5", "custom", path="app/model/project_model_kaggle.pt")
        model = my_model()
        results = model(img)
        detections = results.pandas().xyxy[0]
        # print(detections)
        # assign a unique ID to each object 
        detections['id'] = range(1, len(detections) + 1)
        print(detections , "detections have been printed see")
        color_map = {
            'bike': (0, 255, 0),
            'helmet': (0, 0, 255),
            'no_helmet': (255, 0, 0),
            'number_plate': (255, 255, 0),
            'rider': (0, 255, 255)
        }

        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        # draw bounding boxes on the image
        for index, row in detections.iterrows():
            confidence = row['confidence']
            if confidence < 0.4:
                continue
            x1 = int(row['xmin'])
            y1 = int(row['ymin'])
            x2 = int(row['xmax'])
            y2 = int(row['ymax'])

            class_name = row['name']
            confidence = row['confidence']

            color = color_map.get(class_name, (0, 0, 0))

            # draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            # put text
            cv2.putText(img, f'{class_name} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        is_saved = cv2.imwrite(os.path.join('app/static/images/results/', filename), img)
        # getting the info from the image

        bike_track = []
        rider_track = []
        plate_track = []
        no_helmet_track = []

        all_bikes = []
        all_riders = []

        bike_rider_distance = {}
        bike_plate_distance = {}
        no_helmet_distance = {}

        bike_rider_info = [] # list of tuple (bike_id, rider_id)
        bike_plate_info = [] # list of tuple (bike_id, plate_id)
        no_helmet_info = [] # list of tuple (rider_id, no_helmet_id)

        

        for row in detections.iterrows():
            # part 1
            if row[1]['name'] == 'rider':
                rider_id = row[1]['id']
                print(" ther risder id is : ", rider_id)

                # just keeping record of all riders
                if not rider_id in all_riders:
                    all_riders.append(rider_id)

                # Check if the id exist in rider_track
                print("checking rider id track : ", rider_track)
                if rider_id in rider_track:
                    # print("rider ", rider_id, " already in track")
                    continue

                # get rider coordinates
                rx1 = int(row[1]['xmin'])
                ry1 = int(row[1]['ymin'])
                rx2 = int(row[1]['xmax'])
                ry2 = int(row[1]['ymax'])

                # part 1A
                # looping through the available bikes an associating a rider to the respective bike
                for bike in detections[detections['name'] == 'bike'].iterrows():
                    print(" i thing its here ", bike[1]['id'])
                    bike_id = bike[1]['id']


                    # just recording all bikes
                    if bike_id not in all_bikes:
                        all_bikes.append(bike_id)

                    # bike cordinates
                    bx1 = int(bike[1]['xmin'])
                    by1 = int(bike[1]['ymin'])
                    bx2 = int(bike[1]['xmax'])
                    by2 = int(bike[1]['ymax'])

                    # calculating the distance between each bike and rider
                    # dist = distance(bx1, by1, rx1, ry1)
                    dist = ry2 - by2
                    # print(f"distance from bike: {bike_id} to rider:{rider_id} is {dist}")
                    bike_rider_distance[bike_id] = (rider_id, bike_id, dist)




                # finding min rider with distance to the bike
                min_rider_id = min(bike_rider_distance, key=lambda x: bike_rider_distance[x][2])
                
                rider_id = bike_rider_distance[min_rider_id][0]
                bike_id = bike_rider_distance[min_rider_id][1]
                bike_rider_info.append((bike_id, rider_id))
                # clearing the bike rider distance dict
                bike_rider_distance.clear()


                # part 1B
                # checking if there is any no_helmet case and associating it with a rider
                for no_helmet in detections[detections['name'] == 'no_helmet'].iterrows():
                    no_helmet_id = no_helmet[1]['id']

                    # Check if the id exist in rider_track
                    if no_helmet_id in no_helmet_track:
                        # print("no_helmet ", no_helmet_id, " already in track")
                        continue


                #
                    # Assuming frame dimensions
                    height, width, channels = img2.shape
                    frame_width = width
                    frame_height = height

                    # Get the no-helmet coordinates and clip them
                    nx1 = max(0, min(frame_width, int(no_helmet[1]['xmin'])))
                    ny1 = max(0, min(frame_height, int(no_helmet[1]['ymin'])))
                    nx2 = max(0, min(frame_width, int(no_helmet[1]['xmax'])))
                    ny2 = max(0, min(frame_height, int(no_helmet[1]['ymax'])))
                


                    if not ((rx1 - 5 <= nx1) and (ry1 + 20 >= ny1) and (rx2 + 5 >= nx2) and (ry2 + 5 >= ny2)):
                        # print(" true ....................................................... ri")
                        continue

                    new_ny2 = int(ny2/2)
                    # get the central poing of x and y
                    n_helmet_center_x = (nx1+nx2)/2
                    n_helmet_center_y = (ny1+new_ny2)/2

                    # now calculating the central point euclidean distance between the rider and helmet
                    rider_helmet_dist = distance(rx1, ry1, nx1 , new_ny2)
                    # print(f"distance from rider id :{rider_id} to no_helmet:{no_helmet_id} is {rider_helmet_dist}")

                    # add the rider helmet distance information to no helmet distance
                    no_helmet_distance[no_helmet_id] = (rider_id, no_helmet_id, rider_helmet_dist)

                # get the minimum no helmet distance
                if not no_helmet_distance:
                    # print("no_helmet_distance is empty")
                    continue
                min_no_helmet_id = min(no_helmet_distance, key=lambda x: no_helmet_distance[x][2])

                # append the no helmet minimum id to the no helmet track
                no_helmet_track.append(no_helmet_distance[min_no_helmet_id][1])
                # add the info to the no helmet info list
                no_helmet_info.append((rider_id, no_helmet_distance[min_no_helmet_id][1]))
                # clearing the no helmet distance dict
                no_helmet_distance.clear()




                print(bike_rider_distance)
                # print(f"bike rider info : {bike_rider_info}")

            # part 2
            # handling the assiciation between bike and numberplate
            if row[1]['name'] == 'bike':
                # print(" bike id to be gotten here : ", row[1]['id'])
                bike_id = row[1]['id']

            

                if bike_id in bike_track:
                    # print("bike ", bike_id, " already associated with a number plate")
                    continue

                bx1 = int(row[1]['xmin'])
                by1 = int(row[1]['ymin'])
                bx2 = int(row[1]['xmax'])
                by2 = int(row[1]['ymax'])

                # looping through the available plates and associating a bike to the respective plate
                min_plate_id = None
                for plate in detections[detections['name'] == 'number_plate'].iterrows():
                    plate_id = plate[1]['id']

                    # Check if the id exist in plate_track
                    if plate_id in plate_track:
                        # print("plate ", plate_id, " already associated with a bike")
                        continue

                    px1 = int(plate[1]['xmin'])
                    py1 = int(plate[1]['ymin'])
                    px2 = int(plate[1]['xmax'])
                    py2 = int(plate[1]['ymax'])



                    # calculating the center points of the plate
                    plate_center_x = (px1+px2)/2
                    plate_center_y = (py1+py2)/2

                    # calculating the center points of the bike
                    bike_center_x = (bx1+bx2)/2
                    bike_center_y = (by1+by2)/2

                    # calculating the euclidean distance between the plate and bike central points
                    bp_dist = distance(plate_center_x, plate_center_y, bike_center_x, bike_center_y)
                    # print(f"distance from bike: {bike_id} to plate:{plate_id} is {bp_dist}")
                    bike_plate_distance[plate_id] = (bike_id, plate_id, bp_dist)
                    min_plate_id = min(bike_plate_distance, key=lambda x: bike_plate_distance[x][2])
                # bike_plate_info.append((bike_id, min_plate_id))
                # # plate used
                # plate_track.append(min_plate_id)
                # Ensure that there is at least one plate in the bike_plate_distance dictionary
                if bike_plate_distance:
                    min_plate_id = min(bike_plate_distance, key=lambda x: bike_plate_distance[x][2])
                    bike_plate_info.append((bike_id, min_plate_id))
                    plate_track.append(min_plate_id)
                else:

                    # print(f"No valid plates found for bike {bike_id}")
                    # Optionally, handle the case when no plates are found (e.g., set min_plate_id to None or skip this bike)
                    min_plate_id = None
                    bike_plate_info.append((bike_id, min_plate_id))
                    plate_track.append(min_plate_id)
            

    

        summary_info = []
        # print(bike_rider_info)
        # bike_id_check = []
        for data in bike_rider_info:
            b_id = data[0] # getting the bike id
            r_id = data[1] # getting the rider id
            # print(data, b_id)
            
            n_plate = list(filter(lambda x: x[0] == b_id, bike_plate_info))[0][1] # getting the plate id[0][1]
            
            # print(n_plate)
            no_of_riders = list(filter(lambda x: x[0] == b_id, bike_rider_info  )) # getting the number of riders
            # print(len(no_of_riders))
            no_of_no_helmet = list(filter(lambda x: x[0] == r_id, no_helmet_info)) # getting the number of no helmet riders
            # print(len(no_of_no_helmet))
            # print("bike id : ", b_id)
            detec = detections[detections['id'] == b_id]
            # print(" the bike cordinate is : ", detec)
            # print(b_id)

            if not detec.empty:
                row = detec.iloc[0]
                # print(row)
                height, width = img2.shape[:2]

                x1 = int(row['xmin'] - 20)
                x2 = int(row['xmax'] + 20 )

                y1 = int(20)
                y2 = int(row['ymax']+20)

                x1 = max(0, x1)
                x2 = min(int(row['xmax'] + 20 ), width)
                y1 = int(20)
                y2 = int(row['ymax'])

                # print(" the one on the side " , x1,x2,y1,y2)

                if row['name'] == 'no_helmet':

                    summary_info.append((b_id, len(no_of_riders), n_plate, len(no_of_no_helmet) , x1,x2,y1,y2))
                else:
                    summary_info.append((b_id, len(no_of_riders), n_plate, len(no_of_no_helmet),  x1,x2,y1,y2))

        # print("summary info : ", summary_info)
        summary_df = pd.DataFrame(summary_info, columns=['bike_id', 'number_of_riders', 'plate_id','no_helmet_count', 'x1', 'x2', 'y1', 'y2'])
        # summary_df.head()
        print(summary_df)
        final_info = []

        for index, row in summary_df.iterrows():
            # print(row, "intex is t: ", index)
            if row[3] >= 1:
                print(row)
                # Ensure coordinates are within image bounds
                height, width = img2.shape[:2]

                # x1 = int(row[4] - 20)
                # x2 = int(row[5] + 20 )
                x1 = int(row['x1'] - 20)
                x2 = int(row['x2'] + 20 )

                y1 = int(20)
                y2 = int(height) 

                print("from this info ", x1,x2,y1,y2)
                # Coordinates: [y_start:y_end, x_start:x_end]
                cp_image = img2[y1:y2, x1:x2]
                cv2.rectangle(img2, (x1,y1), (x2,y2), (0,0,255), 2)
            

                if cp_image.size > 0:
                    short_uid = str(uuid.uuid4())[:8]
                    cropped_image_name = f'image_{short_uid}.jpg'
                    cv2.imwrite(os.path.join(violation_folder, cropped_image_name), cp_image)
                    # Generate a UUID and take the first 8 characters
                   
                    violation_info = (row.iloc[0], row.iloc[1], row.iloc[2], row.iloc[3], os.path.join(violation_folder, cropped_image_name))
                    db_bike_id = int(row['bike_id'])
                    db_riders = int(row['number_of_riders'])
                    db_no_helmet = int(row['no_helmet_count'])
                    db_image = os.path.join(violation_folder, cropped_image_name)
                    # adding info to db
                    cursor.execute("insert into violations (bike_id, riders, no_helmet, image) Values (?,?,?,?)", (db_bike_id, db_riders, db_no_helmet, db_image) )

                    print("violation",violation_info)
                else:
                    violation_info = (row.iloc[0], row.iloc[1], row[2], row.iloc[3], None)
                    # adding info to db
                    db_bike_id = int(row['bike_id'])
                    db_riders = int(row['number_of_riders'])
                    db_no_helmet = int(row['no_helmet_count'])
                    db_image = None
                    cursor.execute("insert into violations (bike_id, riders, no_helmet, image) Values (?,?,?,?)", (db_bike_id, db_riders, db_no_helmet, db_image) )

                    # print("no_violation", violation_info)
                    # print(f"Error: Cropped image at index {index} is empty!")
            else:
                no_violation_info = (row.iloc[0], row.iloc[1], row.iloc[2], row.iloc[3], None)
                # adding info to db
                db_bike_id = int(row['bike_id'])
                db_riders = int(row['number_of_riders'])
                db_no_helmet = int(row['no_helmet_count'])
                db_image = None
                cursor.execute("insert into violations (bike_id, riders, no_helmet, image) Values (?,?,?,?)", (db_bike_id, db_riders, db_no_helmet, db_image) )

                # print("no_violation", no_violation_info)



        print(summary_df.head())
        # clossing the db
        conn.commit()
        conn.close()
        # end og getting info from the image
        if img is None:
            return jsonify({'message': 'Image not found!'})
        
        if not is_saved:
            return jsonify({'message': 'Image could not be saved!'})
        # Example response after processing
        return jsonify({'message': f'Image {filename} processed successfully!','filename': filename,  'filepath': '/images/results/' + filename})
        
    # except Exception as e:
    #     print(e)
    #     return jsonify({'message': 'An error occurred while processing the image!' })


@image_routes.route('/processed_result')
def processed_result():
    pass