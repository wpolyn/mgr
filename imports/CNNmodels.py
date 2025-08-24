import cv2

def initialize_detector():
    input_320 = (320, 320)
    input_640 = (640, 640)
    input_480 = (480, 480)
    input_4x3 = (640, 480)

    default_score_threshold = 0.9
    adjusted_score_threshold = 0.8

    resolution_catalogue = {
        1 : "320x320",
        2 : "640x640",
        3 : "480x480",
        4 : "640x480"
    }

    detector_catalogue = {
        1 : "Normal-sized YuNet model (320x320 input)",
        2 : "Small-sized YuNet model (320x320 input)",
        3 : "Standard YuNet model (640x640 input)",
        4 : "int8 quantized YuNet model (640x640 input)",
        #5 : "int8 block-quantized YuNet model (640x640 input)",
        6 : "Normal-sized YuNet model (480x480 input)",
        7 : "Small-sized YuNet model (480x480 input)",
        8 : "Normal-sized YuNet model (640x480 input)",
        9 : "Small-sized YuNet model (640x480 input)"
    }

    while True:
        try:
            for key in resolution_catalogue:
                print(f"{key} : {resolution_catalogue[key]}")
            choice = int(input(f"Please choose the detector input resolution supported by this implementation:\n"))
            if choice == 1:
                input_size = input_320
                score_threshold = default_score_threshold
            elif choice == 2:
                input_size = input_640
                score_threshold = default_score_threshold
            elif choice == 3:
                input_size = input_480
                score_threshold = default_score_threshold
            elif choice == 4:
                input_size = input_4x3
                score_threshold = default_score_threshold
            else:
                print(f"Your input {choice} does not correspond to any resolution. Please modify your input.")
                continue

            if input_size == input_320:
                for key in detector_catalogue:
                    if 1 <= key <= 2: 
                        print(f"{key} : {detector_catalogue[key]}")
                choice = int(input(f"Please choose the detector model by selecting its corresponding number:\n"))
                if choice == 1:
                    model_path = "models/yunet/yunet_n_320_320.onnx"
                    detector_name = "YuNet_n_320x320"
                elif choice == 2:
                    model_path = "models/yunet/yunet_s_320_320.onnx"
                    detector_name = "YuNet_s_320x320"
                else:
                    print(f"Your input {choice} does not correspond to any model. Please modify your input.")
                    continue
            elif input_size == input_640:
                for key in detector_catalogue:
                    if 3 <= key <= 4: 
                        print(f"{key} : {detector_catalogue[key]}")
                choice = int(input(f"Please choose the detector model by selecting its corresponding number:\n"))
                if choice == 3:
                    model_path = "models/yunet/face_detection_yunet_2023mar.onnx"
                    detector_name = "YuNet_640x640"
                elif choice == 4:
                    model_path = "models/yunet/face_detection_yunet_2023mar_int8.onnx"
                    detector_name = "YuNet_int8_640x640"
                    score_threshold = adjusted_score_threshold
                #elif choice == 5:
                    #model_path = "models/yunet/face_detection_yunet_2023mar_int8bq.onnx"
                    #detector_name = "YuNet_int8bq_640x640"
                else:
                    print(f"Your input {choice} does not correspond to any model. Please modify your input.")
                    continue
            elif input_size == input_480:
                for key in detector_catalogue:
                    if 6 <= key <= 7: 
                        print(f"{key} : {detector_catalogue[key]}")
                choice = int(input(f"Please choose the detector model by selecting its corresponding number:\n"))
                if choice == 6:
                    model_path = "models/yunet/yunet_n_fixed_480x480.onnx"
                    detector_name = "YuNet_n_480x480"
                elif choice == 7:
                    model_path = "models/yunet/yunet_s_fixed_480x480.onnx"
                    detector_name = "YuNet_s_480x480"
                else:
                    print(f"Your input {choice} does not correspond to any model. Please modify your input.")
                    continue     
            elif input_size == input_4x3:
                for key in detector_catalogue:
                    if 8 <= key <= 9: 
                        print(f"{key} : {detector_catalogue[key]}")
                choice = int(input(f"Please choose the detector model by selecting its corresponding number:\n"))
                if choice == 8:
                    model_path = "models/yunet/yunet_n_fixed_640x480.onnx"
                    detector_name = "YuNet_n_640x480"
                elif choice == 9:
                    model_path = "models/yunet/yunet_s_fixed_640x480.onnx"
                    detector_name = "YuNet_n_640x480"
                else:
                    print(f"Your input {choice} does not correspond to any model. Please modify your input.")
                    continue
        except ValueError:
            print("Please choose one of the listed numbers.")
            continue
        break
    
    detector = cv2.FaceDetectorYN.create(model_path, "", input_size, score_threshold)

    return detector, detector_name, input_size

def initialize_recognizer():

    recognizer_catalogue = {
        "1" : "Standard SFace model",
        "2" : "int8 quantized SFace model"#,
        #"3" : "int8 block-quantized SFace model"
    }

    for key in recognizer_catalogue:
        print(f"{key} : {recognizer_catalogue[key]}")

    while True:
        try:
            choice = int(input(f"Please choose the recognizer model by selecting its corresponding number:\n"))
            if choice == 1:
                model_path = "models/sface/face_recognition_sface_2021dec.onnx"
                recognizer_name = "SFace"
            elif choice == 2:
                model_path = "models/sface/face_recognition_sface_2021dec_int8.onnx"
                recognizer_name = "SFace_int8"
            #elif choice == 3:
                #model_path = "models/sface/face_recognition_sface_2021dec_int8bq.onnx"
                #recognizer_name = "SFace_int8bq"
            else:
                print(f"Your input {choice} does not correspond to any recognizer model. Please modify your input.")
        except ValueError:
            print("Please choose one of the listed numbers.")
            continue
        break
    
    recognizer = cv2.FaceRecognizerSF.create(model_path, "")

    return recognizer, recognizer_name