import os
import shutil
from googletrans import Translator
import cv2
import argparse
import unidecode
import pandas
import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib

from modules import Preprocess, Detection, OCR, Retrieval, Correction
from tool.config import Config
from tool.utils import natural_keys, visualize, find_highest_score_each_class
import time

parser = argparse.ArgumentParser("Document Extraction")
parser.add_argument("--input", help="Path to single image to be scanned")
parser.add_argument("--output", default="./results", help="Path to output folder")
parser.add_argument("--debug", action="store_true", help="Save every steps for debugging")
# parser.add_argument("--debug", action="store_false", help="Save every steps for debugging")
parser.add_argument("--do_retrieve", action="store_true", help="Whether to retrive information")
# parser.add_argument("--do_retrieve", action="store_false", help="Whether to retrive information")
parser.add_argument("--find_best_rotation", action="store_true",
                    help="Whether to find rotation of document in the image")
# parser.add_argument("--find_best_rotation", action="store_false", help="Whether to find rotation of document in the image")
args = parser.parse_args()


class Pipeline:
    def __init__(self, config):
        self.reader = None
        self.translator = None
        self.referenced_translations = None
        # self.output = args.output
        self.output = "output"
        self.debug = False
        self.do_retrieve = False
        self.find_best_rotation = False
        self.load_config(config)
        self.make_cache_folder()
        self.init_modules()

    def load_config(self, config):
        self.det_weight = config.det_weight
        self.ocr_weight = config.ocr_weight
        self.det_config = config.det_config
        self.ocr_config = config.ocr_config
        self.bert_weight = config.bert_weight
        self.class_mapping = {k: v for v, k in enumerate(config.retr_classes)}
        self.idx_mapping = {v: k for k, v in self.class_mapping.items()}
        self.dictionary_path = config.dictionary_csv
        self.retr_mode = config.retr_mode
        self.correction_mode = config.correction_mode

    def make_cache_folder(self):
        self.cache_folder = os.path.join(self.output, 'cache')
        os.makedirs(self.cache_folder, exist_ok=True)
        self.preprocess_cache = os.path.join(self.cache_folder, "preprocessed.jpg")
        self.detection_cache = os.path.join(self.cache_folder, "detected.jpg")
        self.crop_cache = os.path.join(self.cache_folder, 'image_crops')
        self.final_output = os.path.join(self.output, 'result.jpg')
        self.retr_output = os.path.join(self.output, 'result.txt')
        self.csv_output = os.path.join(self.output, 'result.csv')

    def init_modules(self):
        # self.det_model = Detection(
        #     config_path=self.det_config,
        #     weight_path=self.det_weight)
        self.ocr_model = OCR(
            config_path=self.ocr_config,
            weight_path=self.ocr_weight)
        # self.preproc = Preprocess(
        #     det_model=self.det_model,
        #     ocr_model=self.ocr_model,
        #     find_best_rotation=self.find_best_rotation)

        if self.dictionary_path is not None:
            self.dictionary = {}
            df = pd.read_csv(self.dictionary_path)
            for id, row in df.iterrows():
                self.dictionary[row.text.lower()] = row.lbl
        else:
            self.dictionary = None

        self.correction = Correction(
            dictionary=self.dictionary,
            mode=self.correction_mode)

        if self.do_retrieve:
            self.retrieval = Retrieval(
                self.class_mapping,
                dictionary=self.dictionary,
                mode=self.retr_mode,
                bert_weight=self.bert_weight)

    def craft(self, img):
        # import craft functions
        from craft_text_detector import (
            read_image,
            load_craftnet_model,
            load_refinenet_model,
            get_prediction,
            export_detected_regions,
            export_extra_results,
            empty_cuda_cache
        )

        # set image path and export folder directory
        image = img  # can be filepath, PIL image or numpy array
        output_dir = 'output/cache'

        # read image
        image = read_image(image)

        # load models
        refine_net = load_refinenet_model(cuda=False,weight_path="weights/craft_refiner_CTW1500.pth")
        craft_net = load_craftnet_model(cuda=False,weight_path ="weights/craft_mlt_25k.pth")

        # perform prediction
        prediction_result = get_prediction(
            image=image,
            craft_net=craft_net,
            refine_net=refine_net,
            text_threshold=0.7,
            link_threshold=0.4,
            low_text=0.4,
            cuda=False,
            long_size=1280
        )

        # export detected text regions
        exported_file_paths = export_detected_regions(
            image=image,
            regions=prediction_result["boxes"],
            output_dir=output_dir,
            rectify=True)

        # export heatmap, detection points, box visualization
        export_extra_results(
            image=image,
            regions=prediction_result["boxes"],
            heatmaps=prediction_result["heatmaps"],
            output_dir="output_craft"
        )

        # unload models from gpu
        empty_cuda_cache()
        new_coors = []
        with open("output_craft/image_text_detection.txt", "r") as f:
            coors = [line.strip() for line in f if line.strip() != ""]
            for c in coors:
                c = c.split(",")
                x14 = int(min([c[i] for i in range(0, len(c), 2)]))
                x23 = int(max([c[i] for i in range(0, len(c), 2)]))
                y34 = int(min([c[i] for i in range(1, len(c), 2)]))
                y12 = int(max([c[i] for i in range(1, len(c), 2)]))

                new_coors.append([(x14, y12), (x23, y12), (x23, y34), (x14, y34)])
        new_coors_np = np.squeeze(np.array(new_coors))
        return new_coors_np

    def start(self, img):
        # img1 = self.preproc(img)
        img1 = img
        time_start = time.time()
        boxes = self.craft(img1)
        time_for_detect = time.time()
        print("TIME FOR DETETCT: ", time_for_detect - time_start)
        img_paths = os.listdir(self.crop_cache)
        img_paths.sort(key=natural_keys)
        img_paths = [os.path.join(self.crop_cache, i) for i in img_paths]

        texts = self.ocr_model.predict_folder(img_paths, return_probs=False)
        texts = self.correction(texts, return_score=False)
        time_for_reconize = time.time()
        print("TIME FOR RECONIZE: ", time_for_reconize - time_for_detect)
        if self.do_retrieve:
            preds, probs = self.retrieval(texts)
        else:
            preds, probs = None, None

        visualize(
            img1, boxes, texts,
            img_name=self.final_output,
            class_mapping=self.class_mapping,
            csv_output=self.csv_output,
            labels=preds, probs=probs,
            visualize_best=self.do_retrieve,
        )

        if self.do_retrieve:
            best_score_idx = find_highest_score_each_class(preds, probs, self.class_mapping)
            with open(self.retr_output, 'w') as f:
                for cls, idx in enumerate(best_score_idx):
                    f.write(f"{self.idx_mapping[cls]} : {texts[idx]}\n")
        cv2.waitKey(0)
        time_for_visualize = time.time()
        print("TIME FOR VISUALIZE: ", time_for_visualize - time_for_reconize)

        VietnameseName,Price=self.match_price("output/result.csv")
        time_for_match_price = time.time()
        print("TIME FOR MATCH PRICE: ", time_for_match_price - time_for_visualize)
        return VietnameseName,Price

    def match_price(self, file):
        def export_prize(price):
            kq = ''.join(l for l in price if l.isnumeric())
            if len(kq) < 4: kq += "000"
            return kq

        def has_numbers(s):
            return any(char.isdigit() for char in s)

        def dish_rec(text_size, mean_text_size):
            if abs(text_size - mean_text_size) < 10:
                return True
            else:
                return False

        def price_rec(may_price):
            p = ''.join(l for l in may_price if l.isnumeric())
            if len(may_price) - len(p) < 7 and len(p) < 7:
                if len(p) < 4:
                    p += "000"
                return p
            return False

        def distance_cal(x, y):
            return abs(x[2] - y[2])

        def match(dishes, prices):
            VietnameseName=[]
            Price=[]
            for dish in dishes:
                try:
                    prices2 = [p for p in prices]
                    distances = [distance_cal(dish, price) for price in prices]
                    price = prices2[distances.index(min(distances))]
                    prices2.remove(price)

                    while price[1] < dish[1]:
                        distances = [distance_cal(dish, price) for price in prices2]
                        price = prices2[distances.index(min(distances))]
                        prices2.remove(price)

                except:
                    price = ["NOT GIVEN"]

                VietnameseName.append(dish[0])
                Price.append(price[0])
            return VietnameseName,Price

        data = pd.read_csv(file)
        data.sort_values(by=['x1'], inplace=True)

        coor_dishes = list(zip(data.text, data.width_location, data.height_location))
        text_sizes = data["text_size"].to_list()
        texts = data["text"].to_list()

        def remove_accent(text):
            return unidecode.unidecode(text)

        #
        # for c in coor_dishes:
        #     global ImageName, VietnameseName, Price
        #     text = remove_accent(c[0]).upper()

        # if "COMBO" in text:
        #     # if text != "COMBO 1" and text != "COMBO 2" and text != "COMBO 3" and text != "COMBO 4" and text != "COMBO 5" and text != "COMBO 6" and text != "COMBO 7" and text != "COMBO 8" and text != "COMBO 9" and text != "COMBO 10":
        #     price = export_prize(text)
        #
        #     ImageName.append(f[len(f)-8:])
        #     VietnameseName.append("COMBO")
        #     Price.append(price)
        #     coor_dishes.remove(c)
        #     EnglishName.append("COMBO")

        may_dishes = []
        may_prices = []

        for t in coor_dishes:
            if has_numbers(t[0]):
                may_prices.append(t)
            else:
                if remove_accent(t[0]) != t[0]:
                    may_dishes.append(t)
        mean_text_size = data["text_size"].mean()
        dishes = [(t[0].upper(),
                   t[1],
                   t[2]) for t in may_dishes if
                  dish_rec(data.loc[data["text"] == f"{t[0]}", "text_size"].iloc[0], mean_text_size)]

        prices = [(price_rec(t[0]),
                   t[1],
                   t[2]) for t in may_prices if price_rec(t[0]) is not False]
        # time_for_pre = time.time()
        # print("TIME FOR PRE: ", time_for_pre - start_time)

        VietnameseName,Price=match(dishes, prices)
        return VietnameseName,Price
        # time_for_match = time.time()
        # print("TIME FOR match: ", time_for_match - time_for_pre)


def translate(text):
    try:
        df = pd.read_excel("label.xlsx")
        return df.loc[df["VietnameseName"] == f"{text}", "EnglishName"].iloc[0]
    except:
        translated_name = ''
    if translated_name == '':
        translator = Translator()
        translation = translator.translate(str(text), dest='en', src='vi')
        translated_name = translation.text
        pass
        return translated_name.upper()


def translate_all(df_vie):
    df_dict = pd.read_excel("label.xlsx")
    df_dict=df_dict.drop_duplicates(subset=['VietnameseName'])
    df_dict = df_dict[["VietnameseName", "EnglishName"]]
    df_merge = df_vie.merge(df_dict, on=["VietnameseName"], how="left")
    print(df_merge.columns)

    def convert_nan_value(row):
        if pd.isna(row["EnglishName"]):
            return "UNKNOWN"
        else:
            return row["EnglishName"]

    df_merge["EnglishName"] = df_merge.apply(convert_nan_value, axis=1)
    return df_merge

def run_output(img):
    config = Config('./tool/config/configs.yaml')
    try:
        shutil.rmtree("output/cache/image_crops")
    except:
        pass
    pipeline = Pipeline(config)
    VietnameseName,Price=pipeline.start(img)
    output = {
        # "ImageName": ImageName,
        "VietnameseName": VietnameseName,
        # "EnglishName": EnglishName,
        "Price": Price
    }
    # a = time.time()
    op = pd.DataFrame(output)
    op = translate_all(op)
    pairs = op.values.tolist()
    return pairs

if __name__ == '__main__':
    img = cv2.imread("test_data/001.jpeg")
    print(run_output(img))
