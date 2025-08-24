import cv2

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
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    #initialize a window 
    cv2.namedWindow('frame',cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    return cap, img_src

def vertical_image_processing(image, detector):
    display_size = (240, 320)
    image = cv2.flip(image,1)
    h, w = image.shape[:2] #h=480 w=640
    w1 = h*240//320 #360 #albo 0.75
    x1 = (w-w1)//2 #(640-360)//2=140
    image = image[:, x1:x1+w1] #y 0:480, x 140:140+360
    image = cv2.resize(image, display_size)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    return image, gray, faces

def horizontal_image_processing(image, detector):
    display_size = (320, 240)
    image = cv2.flip(image,1)
    image = cv2.resize(image, display_size)
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