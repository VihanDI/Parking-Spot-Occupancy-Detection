from tensorflow.keras.models import load_model
import numpy as np
import cv2

EMPTY = True
NOT_EMPTY = False

# Loading the trained parking spot classification model
MODEL = load_model("model\parking_slot_classifier.h5")

def empty_or_not(spot_bgr):
    if spot_bgr.shape[0] == 0 or spot_bgr.shape[1] == 0:
        return NOT_EMPTY

    try:
        img_resized = cv2.resize(spot_bgr, (72, 72))
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_array = np.expand_dims(img_normalized, axis=0)

        prediction = MODEL.predict(img_array, verbose=0)[0][0]
        return EMPTY if prediction < 0.6 else NOT_EMPTY
    except Exception as e:
        print("Prediction error:", e)
        return NOT_EMPTY



def get_parking_spots_bboxes(connected_components):
    (totalLabels, label_ids, values, centroid) = connected_components

    slots = []
    coef = 1
    for i in range(1, totalLabels):

        x1 = int(values[i, cv2.CC_STAT_LEFT] * coef)
        y1 = int(values[i, cv2.CC_STAT_TOP] * coef)
        w = int(values[i, cv2.CC_STAT_WIDTH] * coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT] * coef)

        slots.append([x1, y1, w, h])

    return slots