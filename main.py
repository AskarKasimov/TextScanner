# -*- coding: utf-8 -*-

# команда Ducker, кейс от Directum

import cv2
from flask import Flask, request
from pytesseract import pytesseract, image_to_data, Output
import json
from spacy import load
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from libtiff import TIFFfile

# загружаем модель
model = load_model('keras_model.h5')

# путь к Tesseract
pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# создаем приложение
app = Flask(__name__)


# функция для убирания шумов
def filt(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    out_binary = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)[1]
    return image



def task1(name):
    img = filt(cv2.imread(name))
    # формируем и преобразуем словарь с данными по каждому обнаруженнему слову
    data = image_to_data(img, lang='rus', output_type=Output.DICT)
    data["text"] = ' '.join(data["text"]) \
        .replace("_", "") \
        .replace("\n", "") \
        .strip() \
        .split()
    # формируем словарь, который в будущем отправим назад в виде .json файла
    js = dict()
    js['text'] = " ".join(data["text"])
    js['tokens'] = list()
    js['source'] = {"width": img.shape[1], "height": img.shape[0]}
    # цикл для перебора слов и занесением нужной информации в итоговый словарь
    for i in data["text"]:
        js["tokens"].append({"text": i,
                             "position": {
                                 "left": float(data['left'][data["text"].index(i)]),
                                 "top": float(data['top'][data["text"].index(i)]),
                                 "width": float(data['width'][data["text"].index(i)]),
                                 "height": float(data['height'][data["text"].index(i)])}})
        # просчет поля offset
        offset = len(" ".join(data["text"][:data["text"].index(i)]))
        js["tokens"][len(js["tokens"]) - 1]["offset"] = offset if offset == 0 else offset + 1


    return js


def task3(txt):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    # Replace this with the path to your image

    img = cv2.imread(txt)
    image = Image.open(txt).convert('RGB')
    # resize the image to a 224x224 with the same strategy as in TM2:
    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    prediction = list(prediction)
    prediction[0] = list(prediction[0])
    type = "main" if prediction[0].index(max(prediction[0])) == 0 else "other"
    return type



def task4(raw_text):

    NER = load("en_core_web_sm")

    name = {
        'PERSON': 'PERSON',
        'MONEY': 'MONEY',
        'DATE': 'DATE',
        'GPE': 'LOCATION',
        'ORG': 'ORGANIZATION'
    }
    dt = dict()
    dt["facts"] = list()

    s = raw_text[:]
    text1 = NER(raw_text)
    i = {}
    # Идём по списку из найденных сущностей
    for word in text1.ents:
        i.clear()
        # Заполнение 'facts'
        i['text'] = word.text
        i['tag'] = name[str(word.label_)]
        sp = str(word.text).split()
        i['tokens'] = []
        i1 = {}
        # Заполнение 'tokens'
        for j in range(len(sp)):
            i1.clear()
            i1['text'] = sp[j]
            i1['offset'] = s.find(sp[j])
            ind = s.find(sp[j])
            # Находим индекс конеца слова
            while ind < len(s) and s[ind].isalpha():
                ind += 1
            # Замена прочитанного слова нулями
            s = s[:s.find(sp[j])] + '0' * (ind - s.find(sp[j])) + s[ind:]
            pop1 = i1.copy()
            i['tokens'].append(pop1)
        pop = i.copy()
        dt['facts'].append(pop)
        return dt






# функция для задачи №1
@app.route('/1', methods=["POST"])
def upload():
    # читаем и сохраняем файл из POST запроса
    request.files['file'].save('result.tif')
    return json.dumps(task1("result.tif"), ensure_ascii=False, indent=4)

  
# функция для задачи №3
@app.route("/3", methods=["POST"])
def neiro():
    request.files['file'].save("result3.jpg")
    img = cv2.imread("result3.jpg")

    return json.dumps({"source": {"width": img.shape[1], "height": img.shape[0], "type": task3("result3.jpg")}}, ensure_ascii=False, indent=4)



# функция для задачи №4
@app.route("/4", methods=["POST"])
def nlp():
    raw_text = str(request.data)
    raw_text = raw_text[2:-1]
    return json.dumps(task4(raw_text), ensure_ascii=False, indent=4)


# функция для задачи №5
@app.route("/5", methods=["POST"])
def final():
    request.files['file'].save("bigtiff.tif")
    otvet = dict()
    otvet["pages"] = list()
    img = Image.open("C:\\Users\\Student\\PycharmProjects\\PythonProject1\\bigtiff.tif")
    for i in range(TIFFfile('image.tif', mode="r").get_depth()):
        img.seek(i)
        img.save('Block_%s.tif' % (i,))
        d = task1('Block_%s.tif' % (i,))
        ts3 = task3('Block_%s.tif' % (i,))
        if ts3 == "main":
            d["facts"] = task4(d["text"])["facts"]
        d["type"] = ts3
        otvet["pages"].append(d)
    return json.dumps(otvet, ensure_ascii=False)


# вызов приложения в случае если оно основное
if __name__ == "__main__":
    app.run(host="0.0.0.0")
