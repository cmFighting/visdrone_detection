import cv2

# cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)＃frame图像，起点坐标，终点坐标（在这里是x+w,y+h,因为w,h分别是人脸的长宽）颜色，线宽）

# regions = [[232, 385, 13, 31, 0], [234, 391, 8, 17, 1]]


def draw(img_path, regions):
    img = cv2.imread(img_path)
    for x, y, w, h, score in regions:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(score, 'face', (int(w / 2 + x), int(y - h / 5)), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 2, 1)
    cv2.imshow('result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    regions = [[232, 385, 13, 31, 0], [234, 391, 8, 17, 1]]
    region1 = [[232, 385, 13, 31, 0]]
    draw('files/0000002_00005_d_0000014.jpg', region1)
