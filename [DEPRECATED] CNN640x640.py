import cv2

from inits2 import load_registry, initialize_camera, text_properties
input_size = (640, 640)
display_size = (640, 480)
crop_size = 720
dotoctor = cv2.FaceDetectorYN.create("models/yunet/face_detection_yunet_2023mar.onnx", "", input_size)

if __name__ == '__main__':
    #run the functions
    cap, img_src = initialize_camera()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    org, fontFace, fontScale, color, thickness = text_properties()
    ready = False

    while (cap.isOpened()):
        ret, image = cap.read()

        key = cv2.waitKey(1) & 0xFF
        if ret:

            h, w = image.shape[:2]
            x_square = (w - crop_size) // 2
            y_square = 0
            image = image[y_square:y_square+crop_size, x_square:x_square+crop_size]
            image = cv2.resize(image, input_size)
            retval, faces = dotoctor.detect(image)
            #
            if faces is not None:
                for face in faces:
                    x, y, w, h = map(int, face[:4])
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #
            x_display = 0        
            y_display = (display_size[0] - display_size[1]) // 2
            display = image[y_display:y_display+display_size[1],x_display:display_size[0]]
            display = cv2.flip(display,1)
            display = cv2.rotate(display, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imshow('frame', display)
            if key == ord('a') and not ready:
                ready = True
            #click B to exit
            elif key == ord('b'):
                break

    cap.release()
    cv2.destroyAllWindows()