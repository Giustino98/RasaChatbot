# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"
from typing import Any, Text, Dict, List
import time

import cv2
from rasa_sdk import Action, Tracker, FormValidationAction
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher

from pymongo import MongoClient

from rasa_sdk.types import DomainDict

from PIL import Image

import numpy as np


class ValidateShowItem(FormValidationAction):
    def name(self) -> Text:
        return "validate_show_item_form"

    def validate_item(self,
                      slot_value: Any,
                      dispatcher: CollectingDispatcher,
                      tracker: Tracker,
                      DomainDict,
                      ) -> Dict[Text, Any]:
        items = tracker.get_slot("items")
        if len(items) > 0:
            if slot_value in items:
                return {"item": slot_value}
        dispatcher.utter_message(text=f"L'oggetto {slot_value} non è tra quelli individuati dalla webcam.")
        return {"item": None}


class ActionShowItem(Action):

    def name(self) -> Text:
        return "action_show_item"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        item_chosen = tracker.get_slot("item")

        MONGODB_HOST = "localhost"
        MONGODB_PORT = 27017

        client = MongoClient(MONGODB_HOST, MONGODB_PORT)

        db = client['RasaChatbot']
        collection = db.HouseItems

        item_db = collection.find_one({'nome': item_chosen})
        num_items = int((len(item_db) - 2) / 2)
        dispatcher.utter_message(
            text="Perfetto, ti mostro n." + str(num_items) + " opzioni che hai a disposizione.")
        for j in range(0, num_items):
          #  img = Image.open(item_db['image' + str(j + 1)])
          #  img.show(title=item_chosen + str(j + 1))
            link = str(item_db['link' + str(j + 1)])
            dispatcher.utter_message(text="Link n." + str(j + 1) + "= " + link)
            dispatcher.utter_message(image=item_db['image' + str(j + 1)] )

        return []


class ActionStartWebcam(Action):
    whT = 320  # dipende dai pesi e dalla configurazione della rete che sto considerando
    confThreshold = 0.5
    nmsThreshold = 0.3
    classes = [56, 58, 65, 67, 74]

    classesFile = 'coco.names'
    classNames = []

    modelConfiguration = 'C:/Users/giust/PycharmProjects/YoloDemo/yolo_cfg/yolo3.cfg'
    modelWeights = 'C:/Users/giust/PycharmProjects/YoloDemo/yolo_weights/yolo3-320.weights'

    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def name(self) -> Text:
        return "action_start_webcam"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        with open(self.classesFile, 'rt') as f:
            classNames = f.read().rstrip('\n').split('\n')

        actual_time = 0
        initial_time = 0
        frame = 0
        fps = 0

        loop = True

        cap = cv2.VideoCapture(0)

        while loop:
            # Calcola gli fps
            if actual_time - initial_time > 1:
                initial_time = actual_time
                fps = frame
                frame = 0
            else:
                frame = frame + 1
                actual_time = time.time()

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            success, img = cap.read()

            blob = cv2.dnn.blobFromImage(img, 1 / 255, (self.whT, self.whT), [0, 0, 0], 1, crop=False)
            outputs = self.getNetOutput(blob)
            # print(outputs[0].shape)   # (300,85)
            # print(outputs[1].shape)   # (1200,85)
            # print(outputs[2].shape)   # (4800,85)

            objects = self.findObjects(outputs, img, classNames)

            cv2.putText(img, "Fps: " + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            cv2.imshow('Image', img)
            k = cv2.waitKey(1) & 0xFF

            object_detected = False

            if k == ord('q'):
                if len(objects) > 0:
                    for i in range(0, len(objects)):
                        dispatcher.utter_message(text="Oggetto n." + str(i + 1) + ": " + objects[i] + "\n")
                #    dispatcher.utter_message(template="utter_user_will")
                    object_detected = True
                else:
                  #  dispatcher.utter_message(text="Nessun oggetto rilevato. Vuoi ripetere la scansione?")
                    object_detected = False
                loop = False
                cv2.destroyAllWindows()
                cap.release()

        return [SlotSet("items", objects), SlotSet("object_detected", object_detected)]

    def getNetOutput(self, blob):
        self.net.setInput(blob)
        layerNames = self.net.getLayerNames()
        # print(layerNames)
        outputNames = [layerNames[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        outputs = self.net.forward(outputNames)

        return outputs

    def findObjects(self, outputs, img, classnames):
        hT, wT, cT = img.shape
        bbox = []
        classIds = []
        confs = []
        objects = []

        for output in outputs:
            for det in output:
                scores = det[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self.confThreshold:
                    # det[0]: cx, det[1]: cy, det[2]: w, det[3]: h
                    [w, h] = int(det[2] * 1280), int(det[3] * 720)
                    [x, y] = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                    bbox.append([x, y, w, h])
                    classIds.append(classId)
                    confs.append(float(confidence))
        # print(len(bbox))
        indices = cv2.dnn.NMSBoxes(bbox, confs, self.confThreshold, self.nmsThreshold)

        # print(indices)
        for i in indices:
            i = i[0]
            if classIds[i] in self.classes:
                box = bbox[i]
                x, y, w, h = box[0], box[1], box[2], box[3]
                cv2.rectangle(img, (x, y), (x + w, y + w), (255, 0, 255), 2)
                cv2.putText(img, f'{classnames[classIds[i]].upper()} {int(confs[i] * 100)}%', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                objects.append(classnames[classIds[i]])

        return objects
