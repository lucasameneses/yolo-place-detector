from ai.ai_model import load_yolov5_model
from ai.ai_model import detection

from helper.params import Parameters
from helper.general_utils import filter_text

from ai.ocr_model import easyocr_model_load

import cv2


params = Parameters()


if __name__ == "__main__":

    model, labels = load_yolov5_model()
    camera = cv2.VideoCapture(0)
    text_reader = easyocr_model_load()

    while 1:

        ret, frame = camera.read()
        if ret:

            detected, _ = detection(frame, model, labels)
            resulteasyocr = text_reader.readtext(detected)
            text = filter_text(params.rect_size, resulteasyocr, params.region_threshold)

            # save_results(text[-1], "ocr_results.csv", "Detection_Images")
            print(text)
            cv2.imshow("detected", detected)

        if cv2.waitKey(1) & 0xFF == 27:
            cv2.destroyAllWindows()
            break
