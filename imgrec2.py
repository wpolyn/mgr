import os
import pickle
import time
import numpy as np
from PIL import Image
from models import initialize_detector, initialize_recognizer

'''
#TODO
WYODRĘBNIĆ LABELS I IMAGES NA 2 FUNKCJE?
ZAPISYWANIE LABELEK JAKO APPENDOWANIE, NIE NADPISANIE?
'''
def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)] 
    face_samples = [] 
    labels = []
    for image_path in image_paths:
        #color image to grayscale
        image = Image.open(image_path).convert('L')
        #grayscale image into a Numpy array
        image_np = np.array(image, 'uint8')
        #filter out non-jpg
        if os.path.split(image_path)[-1].split(".")[-1] != 'jpg':
            continue
        #retrieve all labels
        image_label = str(os.path.split(image_path)[-1].split("_")[0])
        #detect faces
        faces = detector.detectMultiScale(image_np)
        #crop and store face samples, store all labels
        for (x, y, w, h) in faces:
            face_samples.append(image_np[y:y + h, x:x + w])
            labels.append(image_label)
    return face_samples, labels

if __name__ == '__main__':
    img_src = '/root/imagedb' 
    detector = initialize_detector()
    recognizer = initialize_recognizer()

    #run the function
    print("Creating samples...")
    faces, labels = get_images_and_labels(img_src+'/new/')

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
    #print(ids)

    #save the label_to_id registry in a .pkl file
    print("Saving the registry...")
    registry = 'label_to_id.pkl'
    with open(registry, 'wb') as f:
        pickle.dump(label_to_id, f)
    if os.path.exists(registry) and os.path.getsize(registry) > 0:
        print("Registry saved successfully.") 

    #train the model
    print("Training the model...")
    training_start = time.time()
    recognizer.train(faces, np.array(ids))
    training_end = time.time()
    training_time = int((training_end - training_start)*1000)
    print(f"Training completed successfully in {training_time} ms.")

    #save the model .yml file
    print("Saving the model...")
    recognizer.save(img_src+'/model.yml')
    print("Model saved successfully.")