import gdown
import re
import cv2
import numpy as np
import webcolors
import matplotlib
import matplotlib.pyplot as plt
from modules.detection import draw_bbox
import pandas as pd


def download_weights(id_or_url, cached=None, md5=None, quiet=False):
    if id_or_url.startswith('http'):
        url = id_or_url
    else:
        url = 'https://drive.google.com/uc?id={}'.format(id_or_url)

    return gdown.cached_download(url=url, path=cached, md5=md5, quiet=quiet)


weight_url = {
    "pan_resnet18_default": "1GKs-NnezTc6WN0P_Zw7h6wYzRRZdJFKl",
    "pan_resnet18_sroie19": "1-QvIN0MrP28URQILYvLaF1G1eTx2oh8W",
    "transformerocr_mcocr": "1qpXp_-digz2HPTGY_GPdwstzGLhjC_ot",
    "transformerocr_default_vgg": "13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA",
    "transformerocr_default_resnet50": "12dTOZ9VP7ZVzwQgVvqBWz5JO5RXXW5NY",
    "transformerocr_default_resnet50_fpn": "12dTOZ9VP7ZVzwQgVvqBWz5JO5RXXW5NY",
    "transformerocr_config": "1xQqR9swWNCTLEa0ensPDT0HDBHTke3xT",
    "phobert_mcocr": "1v4GQPg4Jx5FWvqJ-2k9YCxEd6iFdlXXa"
}


def download_pretrained_weights(name, cached=None):
    return download_weights(weight_url[name], cached)


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''

    def atoi(text):
        return int(text) if text.isdigit() else text

    return [atoi(c) for c in re.split(r'(\d+)', text)]


STANDARD_COLORS = [
    'Crimson', 'LawnGreen', 'DeepSkyBlue', 'Gold', 'DarkGrey', 'BlanchedAlmond', 'Bisque',
    'Aquamarine', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'Azure', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'AliceBlue', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


def from_colorname_to_bgr(color):
    rgb_color = webcolors.name_to_rgb(color)
    result = (rgb_color.blue, rgb_color.green, rgb_color.red)
    return result


def standard_to_bgr(list_color_name):
    standard = []
    for i in range(len(list_color_name) - 36):  # -36 used to match the len(obj_list)
        standard.append(from_colorname_to_bgr(list_color_name[i]))
    return standard


color_list = standard_to_bgr(STANDARD_COLORS)


def find_highest_score_each_class(labels, probs, class_mapping):
    best_score = [0] * (len(class_mapping.keys()) - 1)  # exclude NONE class
    best_idx = [-1] * (len(class_mapping.keys()) - 1)  # exclude NONE class
    for i, (label, prob) in enumerate(zip(labels, probs)):
        label_idx = class_mapping[label]
        if label_idx != class_mapping["NONE"]:
            if prob > best_score[label_idx]:
                best_score[label_idx] = prob
                best_idx[label_idx] = i
    return best_idx


def visualize(
        img,
        boxes,
        texts,
        img_name,
        class_mapping,
        csv_output,
        labels=None,
        probs=None,
        visualize_best=False,
):
    """
    Visualize an image with its bouding boxes
    """

    if visualize_best:
        assert labels is not None and probs is not None, "To visualize best, please provide labels and probs"

    dpi = matplotlib.rcParams['figure.dpi']
    # Determine the figures size in inches to fit your image
    height, width, depth = img.shape
    figsize = width / float(dpi), height / float(dpi)

    if visualize_best:
        best_score_idx = find_highest_score_each_class(labels, probs, class_mapping)
    fig, ax = plt.subplots(figsize=figsize)

    # Create a Rectangle patch
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    df_location_and_text = pd.DataFrame(columns=['text', 'width_location', 'height_location'])
    for i in range(len(boxes)):

        box = boxes[i]
        text = texts[i]

        if labels is not None:
            label = labels[i]
            label_idx = class_mapping[label]
        if probs is not None:
            prob = probs[i]
            score = np.round(float(prob), 3)

        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = box
        # todo: tradinh doing here
        x1 = max(0, x1)
        x2 = max(0, x2)
        x3 = max(0, x3)
        x4 = max(0, x4)
        y1 = max(0, y1)
        y2 = max(0, y2)
        y3 = max(0, y3)
        y4 = max(0, y4)

        # todo: visualize in here
        def line(p1, p2):
            A = (p1[1] - p2[1])
            B = (p2[0] - p1[0])
            C = (p1[0] * p2[1] - p2[0] * p1[1])
            return A, B, -C

        def intersection(L1, L2):
            D = L1[0] * L2[1] - L1[1] * L2[0]
            Dx = L1[2] * L2[1] - L1[1] * L2[2]
            Dy = L1[0] * L2[2] - L1[2] * L2[0]
            if D != 0:
                x = Dx / D
                y = Dy / D
                return x, y
            else:
                return False

        L1 = line([x1, y1], [x3, y3])
        L2 = line([x2, y2], [x4, y4])
        R = intersection(L1, L2)
        # 'text', 'weight_location', 'height_location']
        # img = cv2.circle(img, center=(int(min(x1, x2, x3, x4)), int(min(y1, y2, y3, y4))), radius=10, color=(0, 0, 255), thickness=-1)

        df_for_concat = pd.DataFrame(data={'text': [text],
                                           'width_location': [R[0] / width],
                                           'height_location': [R[1] / height],
                                           'text_size': abs(y4 - y1),
                                           'x1': [min(x1, x2, x3, x4)],
                                           'y1': [min(y1, y2, y3, y4)]})
        # df_for_concat = pd.DataFrame(
        #     data={'text': [text], 'mean_x_sig': [(x1 + x4) / 2], 'mean_y_sig': [(y1 + y2) / 2],
        #           'text_size': [abs(y1 - y4)]})
        df_location_and_text = pd.concat([df_location_and_text, df_for_concat])

        box = np.array([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])

        if visualize_best:
            color = color_list[label_idx]
            img = draw_bbox(img, [box], color=color)
            if i in best_score_idx:
                plt_text = f'{text}: {label} | {score}'
                plt.text(x1, y1 - 3, plt_text, color=[i / 255 for i in color][::-1], fontsize=10, weight="bold")
        else:
            color = color_list[0]
            img = draw_bbox(img, [box], color=color)
            plt_text = f'{text}'
            plt.text(x1, y1 - 3, plt_text, color=[i / 255 for i in color][::-1], fontsize=10, weight="bold")

    df_location_and_text.to_csv(csv_output)
    # Display the image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    ax.imshow(img)

    plt.axis('off')
    plt.savefig(img_name, bbox_inches='tight')
    plt.close()
