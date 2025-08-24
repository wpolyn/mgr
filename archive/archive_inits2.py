import cv2
import pickle

def initialize_camera():
    cap = cv2.VideoCapture(0)
    img_src = '/root/imagedb'
    #set parameters
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)
    cap.set(cv2.CAP_PROP_CONTRAST, 32)
    cap.set(cv2.CAP_PROP_SATURATION, 64)
    cap.set(cv2.CAP_PROP_HUE, 10)
    cap.set(cv2.CAP_PROP_GAIN, 1)
    cap.set(cv2.CAP_PROP_EXPOSURE, 2000)
    cap.set(cv2.CAP_PROP_GAMMA, 100)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    #initialize a window 
    cv2.namedWindow('frame',cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    return cap, img_src

def image_processing(image, detector):
    image = cv2.flip(image,1)
    #image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    h, w = image.shape[:2] #h=480 w=640
    w1 = h*240//320 #360 #albo 0.75
    x1 = (w-w1)//2 #(640-360)//2=140
    image = image[:, x1:x1+w1] #y 0:480, x 140:140+360 (360, 480) #1.5 same ratio
    image = cv2.resize(image, (240, 320)) 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    return image, gray, faces

def text_properties():
    org = (10, 50)
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.6
    color = (0, 255, 0)
    thickness = 2
    return org, fontFace, fontScale, color, thickness

#def save registry from img files
"""
    pkl_list = [f for f in os.listdir() if f.endswith('.pkl')]
    if pkl_list:
        pkl_list.sort(key = os.path.getmtime, reverse = True)
        registry = os.path.join('.', pkl_list[0])
"""
#retrieve registry from file
def load_registry():
    #find the most recently updated .pkl file and load it as label_to_id
    print("Loading the registry...")
    try:
        with open('label_to_id.pkl', 'rb') as f:
            label_to_id = pickle.load(f)
        print(f"Registry loaded successfully:\n{label_to_id}")
    except EOFError:
        print("Registry file is empty or corrupted.")
    except FileNotFoundError:
        print("Registry file not found.")
    return label_to_id

#invert registry for easy identification by ID
def invert_registry(label_to_id):
    id_to_label = {ID: label for label, ID in label_to_id.items()}
    return id_to_label