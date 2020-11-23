import cv2
import pytesseract
import numpy as np

# Install opencv from https://stackoverflow.com/a/58991547
# Install pytesseract exe from https://github.com/UB-Mannheim/tesseract/wiki
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def main():
    # https://www.twitch.tv/videos/810717434?t=3h24m54s
    # ROI = image[y1:y2, x1:x2]
    image = cv2.imread('screenread/img_vod.jpg')

    forsen_card_og_vertex = np.float32([[84, 33], [426, 62], [82, 165], [427, 188]])
    forsen_card_warp_vertex = np.float32([[0, 0], [360, 0], [0, 135], [360, 135]])
    forsen_card_warp_matrix = cv2.getPerspectiveTransform(forsen_card_og_vertex, forsen_card_warp_vertex)
    forsen_card = cv2.warpPerspective(image, forsen_card_warp_matrix, (360, 135))

    forsen_name = forsen_card[0:47, 45:358]
    forsen_rank = forsen_card[59:134, 2:181]
    forsen_wins = forsen_card[68:127, 258:357]

    forsen_mu_wr_world = image[49:87, 622:715]
    enemy_mu_wr_world = image[52:87, 1209:1299]
    forsen_mu_wr_personal = image[104:141, 638:732]
    enemy_mu_wr_personal = image[105:142, 1187:1276]
    forsen_stage_wr = image[158:196, 665:758]
    enemy_stage_wr = image[158:196, 1163:1253]

    enemy_card_og_vertex = np.float32([[1499, 64], [1844, 33], [1500, 186], [1842, 166]])
    enemy_card_warp_vertex = np.float32([[0, 0], [360, 0], [0, 135], [360, 135]])
    enemy_card_warp_matrix = cv2.getPerspectiveTransform(enemy_card_og_vertex, enemy_card_warp_vertex)
    enemy_card = cv2.warpPerspective(image, enemy_card_warp_matrix, (360, 135))
    cv2.imshow('enemy_card', enemy_card)
    cv2.setMouseCallback('enemy_card', onMouse)

    enemy_name = enemy_card[2:45, 4:315]
    enemy_rank = enemy_card[60:134, 186:357]
    enemy_wins = enemy_card[73:122, 1:107]


    # cv2.imshow('forsen_name', forsen_name)
    # cv2.imshow('forsen_rank', forsen_rank)
    # cv2.imshow('forsen_wins', forsen_wins)
    # cv2.imshow('forsen_mu_wr_world', forsen_mu_wr_world)
    # cv2.imshow('enemy_mu_wr_world', enemy_mu_wr_world)
    # cv2.imshow('forsen_mu_wr_personal', forsen_mu_wr_personal)
    # cv2.imshow('enemy_mu_wr_personal', enemy_mu_wr_personal)
    # cv2.imshow('forsen_stage_wr', forsen_stage_wr)
    # cv2.imshow('enemy_stage_wr', enemy_stage_wr)

    # cv2.imshow('image', image)
    # cv2.setMouseCallback('image', onMouse)
    cv2.waitKey()

    forsen_name = image[37:110, 152:430]
    forsen_name = cv2.cvtColor(forsen_name, cv2.COLOR_BGR2GRAY)
    forsen_name = cv2.GaussianBlur(forsen_name, (3, 3), 0)
    forsen_name = cv2.threshold(forsen_name, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    for i in [1, 3, 7]:
        data = pytesseract.image_to_string(forsen_name, lang='eng', config='--psm ' + str(i))
        print(f"normal, mode {i}: {data}")


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
