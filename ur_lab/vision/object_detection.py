from matplotlib import pyplot as plt
from ultralytics import YOLOWorld
import cv2
import numpy as np
import os


default_classes = (
    'wooden box', 'box',
    'wooden block', 'block',
    'wooden cube', 'cube',
)

class ObjectDetection:
    def __init__(self, classes=default_classes, min_confidence=0.1):
        self.min_confidence = min_confidence

        self.yolo = YOLOWorld('yolov8x-worldv2')  # largest model
        self.yolo.set_classes(classes)

    def detect_objects(self, im_arr, is_rgb=True, max_detections=5):
        '''

        :param im_arr: b x w x h x 3 numpy array
        :return: a tuple of 3 lists: bboxes, confidences, results
            bboxes and confidences for each image is als oa list, since there may be multiple detections
        '''
        # The model works with bgr!!!
        if is_rgb:
            for i in range(len(im_arr)):
                im_arr[i] = cv2.cvtColor(im_arr[i], cv2.COLOR_RGB2BGR)

        results = self.yolo.predict(im_arr, conf=self.min_confidence, agnostic_nms=True, max_det=max_detections)

        bbox_list = []
        confidence_list = []
        for result in results:
            bboxes = result.boxes.xyxy
            confidences = result.boxes.conf
            bbox_list.append(bboxes)
            confidence_list.append(confidences)

        return bbox_list, confidence_list, results

    def get_annotated_images(self, result):
        return cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    detector = ObjectDetection()

    # image_indices = list(range(1, 8))
    # loaded_images = []
    # for idx in image_indices:
    #     image_path = os.path.join("images_data_merged_hires/images", f'image_{idx}.npy')
    #     if os.path.exists(image_path):
    #         image_array = np.load(image_path)
    #         loaded_images.append(image_array)
    #     else:
    #         print(f"Image {image_path} does not exist.")
    #
    # for r in detector.detect_objects(loaded_images):
    #     im_annotated = detector.get_annotated_images(r[2])
    #     plt.imshow(im_annotated)
    #     plt.show()
    #     ##### TODO: better methods for different types of outputs
    # pass

    # load cropped_im.png:
    im = cv2.imread("cropped_im.png")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # plot od for it
    bboxes, _, results = detector.detect_objects(im)
    results = results[0]
    plt.imshow(results.plot())
    plt.show()