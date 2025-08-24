import cv2
import os
import numpy as np

from imports.MLmodels import initialize_detector, initialize_recognizer
from imports.inits_registry import save_registry

def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)] 
    face_samples = [] 
    labels = []
    for image_path in image_paths:
        #filter out non-jpg
        if os.path.split(image_path)[-1].split(".")[-1] != 'jpg':
            continue
        #retrieve all labels
        image_label = str(os.path.split(image_path)[-1].split("_")[0])
        #retrieve images
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        #detect faces
        faces = detector.detectMultiScale(images)
        #crop and store face samples, store all labels
        for (x, y, w, h) in faces:
            face_samples.append(images[y:y + h, x:x + w])
            labels.append(image_label)
    return face_samples, labels

if __name__ == '__main__':
    img_src = '/root/imagedb' 
    detector, detector_name = initialize_detector()
    recognizer, recognizer_name = initialize_recognizer()

    #run the function
    print("Creating samples...")
    faces, labels = get_images_and_labels(img_src+'/images/')

    #convert unique labels to IDs and save the pairs 
    ids = []
    label_to_id = {}
    current_id = 0
    for label in labels:
        if label not in label_to_id:
            label_to_id[label] = current_id
            current_id += 1
    print(label_to_id)
    ids = [label_to_id[label] for label in labels]
    
    #ssave to pickle
    save_registry(label_to_id)

    #train the model
    print("Training the model...")
    recognizer.train(faces, np.array(ids))
    #save the model .yml file
    print("Saving the model...")
    recognizer.save(img_src+'/model.yml')
    print("Model saved successfully.")