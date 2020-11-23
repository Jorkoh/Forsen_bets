import cv2
import pytesseract
import numpy as np
import imutils

# Install opencv from https://stackoverflow.com/a/58991547
# Install pytesseract exe from https://github.com/UB-Mannheim/tesseract/wiki
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

images = {}


def main():
    # https://www.twitch.tv/videos/810717434?t=3h24m54s

    # ROI = image[y1:y2, x1:x2]

    image = cv2.imread('screenread/img_vod.jpg')
    cv2.imshow('image', image)
    cv2.setMouseCallback('image', onMouse)
    cv2.waitKey()

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
    forsen_card = cv2.warpPerspective(image_post_blur, forsen_card_warp_matrix, (360, 135))
    # add_image('forsen_name', forsen_card[0:42, 45:358])
    # add_image('forsen_rank', forsen_card[59:134, 2:181])
    # add_image('forsen_wins', forsen_card[68:127, 258:357])

    # Enemy card
    enemy_card_og_vertex = np.float32([[1499, 64], [1844, 33], [1500, 186], [1842, 166]])
    enemy_card_warp_vertex = np.float32([[0, 0], [360, 0], [0, 135], [360, 135]])
    enemy_card_warp_matrix = cv2.getPerspectiveTransform(enemy_card_og_vertex, enemy_card_warp_vertex)
    enemy_card = cv2.warpPerspective(image_post_blur, enemy_card_warp_matrix, (360, 135))
    # add_image('enemy_name', enemy_card[2:44, 4:315])
    # add_image('enemy_rank', enemy_card[60:134, 186:357])
    # add_image('enemy_wins', enemy_card[73:122, 1:107])

    # Win rates
    # add_image('forsen_mu_wr_world', image_red_channel[49:87, 622:715])
    # add_image('enemy_mu_wr_world', image[52:87, 1209:1299])
    # add_image('forsen_mu_wr_personal', image_red_channel[104:141, 638:732])
    # add_image('enemy_mu_wr_personal', image[105:142, 1187:1276])
    # add_image('forsen_stage_wr', image_red_channel[158:196, 665:758])
    # add_image('enemy_stage_wr', image[158:196, 1163:1253])

    # Prowess
    # add_image('forsen_prowess', image_red_channel[401:438, 98:250])
    # add_image('enemy_prowess', image[402:435, 1660:1822])

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
    forsen_previous__warp_vertex = np.float32([[0, 0], [600, 0], [0, 42], [600, 42]])
    forsen_previous__warp_matrix = cv2.getPerspectiveTransform(forsen_previous_og_vertex, forsen_previous__warp_vertex)
    add_image('forsen_first_stat_letter', cv2.warpPerspective(image[895:930, 115:624], stat_warp_matrix, (600, 42)))

    transcribe_images()
    print_transcriptions()
    show_images()
    cv2.waitKey()


def show_images():
    for image_name in images:
        cv2.imshow(image_name, images[image_name][0])


def transcribe_images():
    for image_name in images:
        # 1, 3, 7? https://wilsonmar.github.io/tesseract/
        images[image_name][1] = pytesseract.image_to_string(images[image_name][0], config='--psm 7')


def print_transcriptions():
    for image_name in images:
        print(f"{image_name}: {images[image_name][1]}")


def add_image(title, image):
    image = imutils.resize(image, width=400)
    images[title] = [image, '']


first_x = -1
first_y = -1


def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # print(f"x: {x}, y: {y}")
        global first_x
        global first_y
        if first_x == -1:
            first_x = x
            first_y = y
        else:
            print(f"[{first_y}:{y}, {first_x}:{x}]")
            first_x = -1
            first_y = -1


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
    main()
