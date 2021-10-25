# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


from typing import Any, Text, Dict, List
import time

import cv2
from rasa_sdk import Action, Tracker, FormValidationAction
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher

from pymongo import MongoClient

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image

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
        items_not_detected = tracker.get_slot("items_not_detected")
        if len(items) > 0:
            if slot_value in items:
                return {"item": slot_value}
        if len(items_not_detected) > 0:
            if slot_value in items_not_detected:
                return {"item": slot_value}
        dispatcher.utter_message(text=f"L'oggetto {slot_value} non è tra quelli suggeriti.")
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
            dispatcher.utter_message(image=item_db['image' + str(j + 1)])

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

    # th architecture to use
    arch = 'resnet18'

    # load the pre-trained weights
    model_file = '%s_places365.pth.tar' % arch

    model = models.__dict__[arch](num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()

    # load the image transformer
    centre_crop = trn.Compose([
        trn.Resize((256, 256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # load the class label
    file_name = 'categories_places365.txt'

    def name(self) -> Text:
        return "action_start_webcam"

    async def run(
            self, dispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        # custom behavior

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
                    # for i in range(0, len(objects)):
                    #    dispatcher.utter_message(text="Oggetto n." + str(i + 1) + ": " + objects[i] + "\n")
                    #    dispatcher.utter_message(template="utter_user_will")
                    object_detected = True
                else:
                    #  dispatcher.utter_message(text="Nessun oggetto rilevato. Vuoi ripetere la scansione?")
                    object_detected = False
                loop = False
                cv2.destroyAllWindows()
                cap.release()

        classes = list()
        with open(self.file_name) as class_file:
            for line in class_file:
                classes.append(line.strip().split(' ')[0][3:])
        classes = tuple(classes)

        # load the test image
        #   img_name = 'DemoPictures/diningroom_demo.jpg'

        #  img = Image.open(img_name)
        cv2_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im)
        input_img = V(self.centre_crop(pil_im).unsqueeze(0))

        # forward pass
        logit = self.model.forward(input_img)
        h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(0, True)

        houseroom_indexes = [203, 215, 121, 45, 52]
        probs_temp = []
        idx_temp = []
        for i in range(0, len(probs)):
            if idx[i] in houseroom_indexes:
                probs_temp.append(probs[i])
                idx_temp.append(idx[i])

        idx = idx_temp
        room = classes[idx[0]]
        #  dispatcher.utter_message(text="Vedo che sei nella stanza: " + classes[
        #     idx[0]] + ". Sei interessato ad acquistare uno dei seguenti oggetti?")

        return [SlotSet("items", objects), SlotSet("object_detected", object_detected), SlotSet("room", room)]

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


class UpdateObjects(Action):

    def name(self) -> Text:
        return "action_update_objects"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        items = tracker.get_slot("items")
        object_detected = tracker.get_slot("object_detected")
        if object_detected:
            for i in range(0, len(items)):
                dispatcher.utter_message(text="Oggetto n." + str(i + 1) + ": " + items[i] + "\n")
                dispatcher.utter_message(template="utter_user_will")
        else:
            dispatcher.utter_message(text="Nessun oggetto rilevato. Vuoi ripetere la scansione?")

        return []


class NewObjects(Action):

    def name(self) -> Text:
        return "action_new_objects"

    async def run(
            self, dispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        # custom behavior
        room = tracker.get_slot("room")
        items = tracker.get_slot("items")

        MONGODB_HOST = "localhost"
        MONGODB_PORT = 27017

        client = MongoClient(MONGODB_HOST, MONGODB_PORT)

        db = client['RasaChatbot']
        collection = db.RoomItems

        item_db = collection.find_one({'room': room})

        dispatcher.utter_message(text="Vedo che sei nella stanza: " + str(
            room) + ". Potresti essere interessato all'acquisto di uno dei seguenti oggetti.")
        items_not_detected = []
        for i in range(0, len(item_db) - 2):
            if item_db['item' + str(i + 1)] not in items:
                dispatcher.utter_message(text="Oggetto nr." + str(i + 1) + ": " + item_db['item' + str(i + 1)])
                items_not_detected.append(item_db['item' + str(i + 1)])

        return [SlotSet("items_not_detected", items_not_detected)]
