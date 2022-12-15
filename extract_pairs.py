from googletrans import Translator
import easyocr
import cv2
import re
from fuzzywuzzy import process
import numpy as np
import time
import argparse
from PIL import Image
import io
import random
import datetime
import string

class Extractor:
  def __init__(self, lang_list=['vi', 'en'], referenced_translations=None):
    self.reader = easyocr.Reader(['vi', 'en'], gpu=False)
    # self.reader = easy_reader
    self.translator = Translator()
    self.referenced_translations = {}
    if referenced_translations:
      labels_df = pd.read_csv(referenced_translations)
      for index, row in labels_df.iterrows():
          self.referenced_translations[row['VietnameseName']] = row['EnglishName']

  def draw_bbs(self, image, horizontal_list):
    color = (255, 0, 0) 
    thickness = 2

    for box in horizontal_list[0]:
      image = cv2.rectangle(image, (box[0], box[2]), (box[1], box[3]), color, thickness) 
    
    return image
  
  def in_one_line(self, coor_1, coor_2):
    coor_1_y_min = coor_1[0][1]
    coor_1_y_max = coor_1[2][1]
    coor_2_y_min = coor_2[0][1]
    coor_2_y_max = coor_2[2][1]
    coor_2_y_center = (coor_2_y_min + coor_2_y_max)/2
    if coor_2_y_center > coor_1_y_min and coor_2_y_center < coor_1_y_max:
      return True
    return False
  
  def extract_lines(self, result):
    patient = 2
    lines = []
    boxes = [box[0] for box in result]
    i = 0
    while i < len(boxes):
      line = [result[i]]
      first_box_id = i
      try:
        while i+1 < len(boxes) and self.in_one_line(boxes[first_box_id], boxes[i+1]):
          line.append(result[i+1])
          i += 1
      except Exception as e:
        print(f'Exception {e}')

      lines.append(line)
      i += 1

    return lines
  
  def filename_gen(self):
    basename = ''.join(random.choice(string.ascii_lowercase) for i in range(4))
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    return "".join([basename, suffix])+".jpg"  # e.g. 'mylogfile_120508_171442'

  def post_process(self, price):
    if price == 'miễnphí':
      return 'Free'
    else:
      if len(price) < 4:
        return price + '000'
      return price

  def extract_menu(self, input):
    if isinstance(input, str):
      ## Detect + OCR ##
      image_path = input
      image = cv2.imread(image_path)
      result = list(self.reader.readtext(image))
      
      ## Extract lines ##
      lines = self.extract_lines(result)

    elif isinstance(input, list):
      lines = input
    
    elif isinstance(input, bytes):
      result = list(self.reader.readtext(input))
      
      ## Extract lines ##
      lines = self.extract_lines(result)

    ## Extract pairs ##
    pairs = []
    # price_pattern = r"(\d{1,3}k(\/\w{1,3})?)|((\d\s?){1,3}[\.\,](\d\s?){3}\s?(d|đ|vnđ|vnd)?)|miễn\sphí"
    price_pattern = r"(\d{2,3}k?)|((\d\s?){1,3}[\.\,](\d\s?){3}\s?)|miễn\sphí"

    '''
    Referenced Translations Structure:

    referenced_translations = {
      'vi_1': 'en_1',
      'vi_2': 'en_2'
    }

    '''
    referenced_names = list(self.referenced_translations.keys())

    for line in lines:
      i = 0
      try:
        while True:
          food_name = line[i][1]
          while not (re.search(price_pattern, food_name.lower().replace('O','0').replace('o','0')) or re.match(price_pattern, line[i+1][1].lower().replace('O','0').replace('o','0'))):
            food_name += ' ' + line[i+1][1]
            # print(line[i+1][1])
            i += 1
            if i + 1 >= len(line):
              break

          if re.search(price_pattern, food_name.lower().replace('O','0').replace('o','0')):
            price = re.search(price_pattern, food_name.lower().replace('O','0').replace('o','0'))
            # print(f'case 1: price = {price}')
            # print(f'foodname: foodname = {food_name[:-len(price.group())]}')
            food_name = food_name[:-len(price.group())]
          else:
            if i + 1 >= len(line):
              break
            price = re.match(price_pattern, line[i+1][1].lower().replace('O','0').replace('o','0'))
            # print(f'case 2: price = {price}')
            # print(f'foodname: foodname = {food_name}')
            
          if price:
            price = price.group().replace(' ', '').replace('.', '').replace('k', '000').replace('O','0').replace('o','0')
            price = self.post_process(price)

            ## Translation ##
            translated_name = ''

            best_match = process.extractOne(food_name, referenced_names)
            if best_match and best_match[1] >= 90:
              try:
                translated_name = self.referenced_translations[best_match[0]].strip()
              except:
                translated_name = ''
              
            if translated_name == '':
              translation = self.translator.translate(food_name, dest='en', src='vi')
              translated_name = translation.text
              pass

            if price[0] != '0':
              pairs.append((food_name.upper(), price, translated_name))
          i += 2
      except Exception as E:
        # print(f'Exception: {E}')
        pass

    return pairs

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--input_file", help="input file path")
  args = parser.parse_args()

  import pandas as pd

  labels_df = pd.read_csv('label.csv')
  referenced_translations = {}
  for index, row in labels_df.iterrows():
      referenced_translations[row['VietnameseName']] = row['EnglishName']

  extractor = Extractor('label.csv')
  start_time = time.time()
  pairs = extractor.extract_menu(args.input_file)
  print(pairs)
  print(f'Elapsed time: {time.time() - start_time}')