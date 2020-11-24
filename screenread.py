import os
import cv2
import pytesseract
import numpy as np
import imutils
import subprocess
from datetime import datetime
import matplotlib.pyplot as plt

# Install opencv from https://stackoverflow.com/a/58991547
# Install pytesseract exe from https://github.com/UB-Mannheim/tesseract/wiki
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

images = {}


def scrape_stream():
    streamlink = subprocess.Popen("streamlink \"https://www.twitch.tv/forsen\" best -O", stdout=subprocess.PIPE)
    ffmpeg = subprocess.Popen("ffmpeg -i pipe:0 -r 0.25 -pix_fmt bgr24 -vcodec rawvideo -an -sn -f image2pipe pipe:1",
                              stdin=streamlink.stdout, stdout=subprocess.PIPE, bufsize=1920 * 1080 * 3)

    cooldown = 0
    while True:
        raw_image = ffmpeg.stdout.read(1920 * 1080 * 3)
        image = np.fromstring(raw_image, dtype='uint8')  # convert read bytes to np
        image = image.reshape((1080, 1920, 3))

        # skip those right after the card has been detected to avoid detecting it multiple times
        if cooldown > 0:
            cooldown -= 1
            continue

        region = cv2.threshold(cv2.GaussianBlur(cv2.inRange(cv2.cvtColor(
            image[49:86, 786:1132], cv2.COLOR_BGR2GRAY), 190, 230), (3, 3), 0), 0, 255, cv2.THRESH_BINARY_INV)[1]
        text = pytesseract.image_to_string(region, config='--psm 7')

        if text == "Match-Up Win Rate (World)\n\f":
            cooldown = 8
            print("DETECTED CARD")
            time = datetime.now().strftime("%H_%M_%S")
            cv2.imwrite(f"images/img_alt_{time}.jpg", image)
            print("SAVED CARD")

            read_image(image)


def read_image(image):
    # ROI = image[y1:y2, x1:x2]

    # cv2.imshow('image', image)
    # cv2.setMouseCallback('image', onMouse)
    # cv2.waitKey()

    image_red_channel = image[:, :, 2]

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image_red_channel = cv2.filter2D(image_red_channel, -1, kernel)
    image_red_channel = cv2.filter2D(image_red_channel, -1, kernel)
    image_red_channel = cv2.threshold(image_red_channel, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    image_red_channel = cv2.GaussianBlur(image_red_channel, (3, 3), 0)

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_pre_blur = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    image_post_blur = cv2.GaussianBlur(image_pre_blur, (3, 3), 0)

    # Forsen card
    forsen_card_og_vertex = np.float32([[84, 33], [426, 62], [82, 165], [427, 188]])
    forsen_card_warp_vertex = np.float32([[0, 0], [360, 0], [0, 135], [360, 135]])
    forsen_card_warp_matrix = cv2.getPerspectiveTransform(forsen_card_og_vertex, forsen_card_warp_vertex)
    forsen_card = cv2.warpPerspective(image, forsen_card_warp_matrix, (360, 135))

    add_image('forsen_name', gray_threshold_blur(forsen_card[0:42, 45:358]))
    add_image('forsen_wins', gray_threshold_blur(forsen_card[68:127, 258:357]))
    forsen_rank = rank_detection(forsen_card[59:134, 2:181])
    print(f"Forsen rank: {forsen_rank}")

    # Enemy card
    enemy_card_og_vertex = np.float32([[1499, 64], [1844, 33], [1500, 186], [1842, 166]])
    enemy_card_warp_vertex = np.float32([[0, 0], [360, 0], [0, 135], [360, 135]])
    enemy_card_warp_matrix = cv2.getPerspectiveTransform(enemy_card_og_vertex, enemy_card_warp_vertex)
    enemy_card = cv2.warpPerspective(image, enemy_card_warp_matrix, (360, 135))

    # TODO: white threshold this?
    add_image('enemy_name', gray_threshold_blur(enemy_card[2:44, 4:315]))
    add_image('enemy_wins', gray_threshold_blur(enemy_card[73:122, 1:107]))
    enemy_rank = rank_detection(enemy_card[60:134, 186:357])
    print(f"Enemy rank: {enemy_rank}")

    # Win rates
    add_image('forsen_mu_wr_world', threshold_red(image[49:87, 622:715]))
    add_image('forsen_mu_wr_personal', threshold_red(image[104:141, 638:732]))
    add_image('forsen_stage_wr', threshold_red(image[158:196, 665:758]))
    add_image('enemy_mu_wr_world', threshold_blue(image[52:87, 1209:1299]))
    add_image('enemy_mu_wr_personal', threshold_blue(image[105:142, 1187:1276]))
    add_image('enemy_stage_wr', threshold_blue(image[158:196, 1163:1253]))

    # Prowess
    add_image('forsen_prowess', threshold_red(image[401:438, 98:250]))
    add_image('enemy_prowess', threshold_blue(image[402:435, 1660:1822]))

    # Top stats
    stat_og_vertex = np.float32([[15, 0], [60, 0], [0, 50], [50, 50]])
    stat_warp_vertex = np.float32([[0, 0], [50, 0], [0, 50], [50, 50]])
    stat_warp_matrix = cv2.getPerspectiveTransform(stat_og_vertex, stat_warp_vertex)
    # add_image('forsen_first_stat_letter', cv2.warpPerspective(image_pre_blur[526:570, 105:200], stat_warp_matrix, (86, 44)))
    # add_image('forsen_second_stat_letter', cv2.warpPerspective(image_pre_blur[588:632, 155:250], stat_warp_matrix, (86, 44)))
    # add_image('forsen_third_stat_letter', cv2.warpPerspective(image_pre_blur[648:692, 200:295], stat_warp_matrix, (86, 44)))
    # add_image('enemy_first_stat_letter', cv2.warpPerspective(image_pre_blur[525:569, 1715:1810], stat_warp_matrix, (86, 44)))
    # add_image('enemy_second_stat_letter', cv2.warpPerspective(image_pre_blur[589:633, 1675:1770], stat_warp_matrix, (86, 44)))
    # add_image('enemy_third_stat_letter', cv2.warpPerspective(image_pre_blur[650:694, 1625:1720], stat_warp_matrix, (86, 44)))
    gray_filtered = cv2.threshold(cv2.GaussianBlur(cv2.inRange(image_gray, 190, 230), (3, 3), 0), 0, 255, cv2.THRESH_BINARY_INV)[1]
    # add_image('forsen_first_stat_name', gray_filtered[524:575, 237:473])
    # add_image('forsen_second_stat_name', gray_filtered[587:638, 282:494])
    # add_image('forsen_third_stat_name', gray_filtered[648:697, 330:498])
    # add_image('enemy_first_stat_name', gray_filtered[523:572, 1449:1684])
    # add_image('enemy_second_stat_name', gray_filtered[585:635, 1460:1637])
    # add_image('enemy_third_stat_name', gray_filtered[646:698, 1399:1591])

    # Previous matches
    forsen_previous_og_vertex = np.float32([[119, 902], [622, 872], [119, 944], [622, 909]])
    forsen_previous_warp_vertex = np.float32([[0, 0], [600, 0], [0, 42], [600, 42]])
    forsen_previous_warp_matrix = cv2.getPerspectiveTransform(forsen_previous_og_vertex, forsen_previous_warp_vertex)
    # add_image('forsen_previous', cv2.warpPerspective(image[895:930, 115:624], forsen_previous_warp_matrix, (600, 42)))

    transcribe_images()
    print_transcriptions()
    # show_images()
    cv2.waitKey()


def gray_threshold_blur(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return image


def threshold_blue(image):
    # cv2.imshow("1", image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # cv2.imshow("2", image)
    mask = cv2.inRange(image, (102, 175, 180), (107, 220, 255))
    # cv2.imshow("3", mask)
    image = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV)[1]
    # cv2.imshow("4", image)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    # cv2.imshow("5", image)
    return image


def threshold_red(image):
    # cv2.imshow("1", image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # cv2.imshow("2", image)
    mask1 = cv2.inRange(image, (176, 230, 200), (181, 255, 255))
    mask2 = cv2.inRange(image, (0, 230, 200), (2, 255, 255))
    mask = cv2.bitwise_or(mask1, mask2)
    # cv2.imshow("3", mask)
    image = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV)[1]
    # cv2.imshow("4", image)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    # cv2.imshow("5", image)
    return image


rank_matcher_data = {}


def prepare_rank_matcher_data():
    for rank in range(1, 36):
        rank_template = cv2.cvtColor(cv2.imread(f"rank_images/{rank}.png"), cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        rank_matcher_data[rank] = sift.detectAndCompute(rank_template, None)


def rank_detection(rank_card):
    rank_card = cv2.cvtColor(rank_card, cv2.COLOR_BGR2GRAY)

    highest_match = (0, 0)
    for rank in range(1, 36):
        sift = cv2.SIFT_create()

        kp1, des1 = sift.detectAndCompute(rank_card, None)
        kp2, des2 = rank_matcher_data[rank]

        flann_index_kdtree = 1
        index_params = dict(algorithm=flann_index_kdtree, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        good_matches = 0
        # ratio test as per Lowe's paper
        for k, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                good_matches += 1

        if good_matches > highest_match[1]:
            highest_match = (rank, good_matches)
    return highest_match[0]


def show_images():
    for image_name in images:
        cv2.imshow(image_name, images[image_name][0])


def transcribe_images():
    for image_name in images:
        # 1, 3, 7? https://wilsonmar.github.io/tesseract/
        images[image_name][1] = pytesseract.image_to_string(images[image_name][0], config='--psm 7')[:-2]


def print_transcriptions():
    for image_name in images:
        print(f"{image_name}: {images[image_name][1]}")


def add_image(title, image):
    image = imutils.resize(image, width=400)
    images[title] = [image, '']


first_x = -1
first_y = -1


def on_mouse_show_roi(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global first_x
        global first_y
        if first_x == -1:
            first_x = x
            first_y = y
        else:
            print(f"[{first_y}:{y}, {first_x}:{x}]")
            first_x = -1
            first_y = -1


def on_mouse_show_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"x: {x}, y: {y}")


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


if __name__ == "__main__":
    prepare_rank_matcher_data()
    # scrape_stream()
    for image_file in os.scandir("images"):
        read_image(cv2.imread(image_file.path))
