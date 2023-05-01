#Import the OpenCV library
import cv2
import numpy as np
import traceback
import winsound
from scipy import stats

############################################################

def deseneaza_dreptunghi(img, x, y, width, height, left_right):
    (x,y,w,h) = x, y, width, height;
    rectangle_color = RECTANGLE_COLOR
    if left_right == "left":
        rectangle_color = (125, 245, 130)

    cv2.rectangle(
        img,
        (x-10, y-20),
        (x + w+10, y + h+20),
        rectangle_color,
        2)
    return img
def deseneaza_linii(img, lines, color="GREEN"):
    # generate start, end for each line
    line_pairs = []
    for index in range(len(lines)):
        if(index > 0):
            start = lines[index-1]
            end = lines[index]
            line_pairs.append((start, end))

    # draw each line onto image
    for line_x, line_y in line_pairs:

        img = deseneaza_linie(img, line_x, line_y, color)

    # return image
    return img

def deseneaza_linie(img, line_x, line_y, color="GREEN"):
    x1, x2 = line_x
    xb = int(x2)
    y1, y2 = line_y
    yb = int(y2)
    if(color=="GREEN"): color = (0, 255, 0)
    if(color=="BLUE"): color = (255, 0, 0)
    if(color=="RED"): color = (0, 0, 255)
    if(color=="Other"): color = (150, 200, 255)
    cv2.line(img, (x1, xb), (y1, yb), color, 2)
    return img

def calculate_max_contrast_pixel(img_gray, x, y, h, top_values_to_consider=3, search_width = 10):
    a = int(y+h)
    y1 = int(y)
    s = x-search_width//2
    s1 = x+search_width//2
    columns = img_gray[y1:a, s:s1]
    column_average = columns.mean(axis=1)
    gradient = np.gradient(column_average, 3)
    gradient = np.absolute(gradient) # abs gradient value
    max_indicies = np.argpartition(gradient, -top_values_to_consider)[-top_values_to_consider:] # indicies of the top 5 values
    max_values = gradient[max_indicies]
    if(max_values.sum() < top_values_to_consider):
        return None # return none if no large gradient exists - probably no shoulder in the range
    weighted_indicies = (max_indicies * max_values)
    weighted_average_index = weighted_indicies.sum() // max_values.sum()
    try:
        index = int(weighted_average_index)
        index = y + index
    except:
        index = 1
    return index

def gaseste_umeri(img_gray, x, y, width, height, direction, x_scale=0.75, y_scale=0.75):
    x_face, y_face, w_face, h_face = x, y, width, height # define face components

    # define shoulder box componenets
    w = int(x_scale * w_face)
    h = int(y_scale * h_face)
    y = y_face + h_face * 3//4 # half way down head position
    if(direction == "right"): x = x_face + w_face - w // 10 # right end of the face box
    if(direction == "left"): x = x_face - w + w // 10  # w to the left of the start of face box
    rectangle = (x, y, w, h);

    # calculate position of shoulder in each x strip
    x_positions = []
    y_positions = []
    for delta_x in range(w):
        this_x = x + delta_x
        this_y = calculate_max_contrast_pixel(img_gray, this_x, y, h)
        if(this_y is None):
            continue # dont add if no clear best value
        x_positions.append(this_x);
        y_positions.append(this_y);

    # extract line from positions
    #line = [(x_positions[5], y_positions[5]), (x_positions[-10], y_positions[-10])];
    lines = [];
    for index in range(len(x_positions)):
        lines.append((x_positions[index], y_positions[index]))

    # extract line of best fit from lines
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_positions, y_positions)
    line_y0 = int(x_positions[0] * slope + intercept)
    line_y1 = int(x_positions[-1] * slope + intercept)
    line = [(x_positions[0], line_y0), (x_positions[-1], line_y1)]

    # decide on value
    #value = intercept;
    value = np.array([line[0][1], line[1][1]]).mean()

    # return rectangle and positions
    return line, lines, rectangle, value

def gaseste_ochi(img_gray):
    eyes = eye_detector.detectMultiScale(img_gray)
    #print("ochii = " + str(eyes))
    return eyes

history_dict = dict({
    "RIGHT" : [],
    "LEFT" : [],
})

def calculeaza_info_postura(left_eye_x, right_eye_x, left_shoulder_x, right_shoulder_x):
    #print("left_shoulder_x = " + str(left_shoulder_x))
    #print("Acces save posture, left_eye_x " + str(left_eye_x) + " right_eye_x = " + str(right_eye_x))
    #print("left_shoulder_x = " + str(left_shoulder_x) + " right_shoulder_x = " + str(right_shoulder_x))
    l_ex = left_eye_x
    r_ex = right_eye_x
    l1_sx,l2_sx = left_shoulder_x
    l11_sx, _ = l1_sx
    l21_sx,_ = l2_sx
    l_sx = (l11_sx + l21_sx)
    r1_sx,r2_sx= right_shoulder_x
    r11_sx, _ = r1_sx
    r21_sx, _ = r2_sx
    r_sx = (r11_sx + r21_sx)
    dist_l_e_s = l_ex - l_sx // 2
    dist_r_e_s = r_ex - r_sx // 2
    return dist_l_e_s, -dist_r_e_s

def find_and_display(capture):
    #Retrieve the latest image from the webcam
    pos_saved = False
    key_presed = False
    d_l_es = 0
    d_r_es = 0
    var_normalizare = 0
    while True:
        rc, img = capture.read()

        key = cv2.waitKey(5)
        #print("key = " + str(key))
        if key == 27:
            break

        if key == 32:
            key_presed = True
            pos_saved = False

        if key > -1:
            print("tasta = " + str(key))

        #noiseless_image_colored = cv2.fastNlMeansDenoisingColored(img, None, 10, 20, 4, 10)
        #cv2.imshow('webcam_feed2', noiseless_image_colored)

        # detection expects grayscale image, convert to grayscale to run basic algorithm of detecting shoulder
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.equalizeHist(img_gray)
        t = 130  # threshold: tune this number to your needs

        # Threshold
        ret, thresh = cv2.threshold(img_gray, t, 255, cv2.THRESH_BINARY_INV)
        # kernel = np.ones((9,9),np.uint8)
        # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        cv2.imshow("thresh", thresh)
        # Canny Edge detection
        canny = cv2.Canny(thresh, 0, 100)
        cv2.imshow("canny", canny)


        #cv2.imshow('webcam_feed2', img_gray)

        try:
            # extract the face
            #face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            face = face_detector.detectMultiScale(img_gray, 1.3, 5)

            # detect shoulders
            if len(face):
                #print("FATAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" + str(face))
                for (x, y, width, height) in face:
                    #print("test " + str(x)  + ", " + str(y) + ", " + str(width) +  ", " + str(height))
                    right_shoulder_line, right_shoulder_lines, right_shoulder_rectangle, right_shoulder_value = gaseste_umeri(canny, x, y, width, height, "right")
                    left_shoulder_line, left_shoulder_lines, left_shoulder_rectangle, left_shoulder_value = gaseste_umeri(canny, x, y, width, height, "left")


                eyes = gaseste_ochi(img_gray)
                # plot face onto image
                img = deseneaza_dreptunghi(img, x, y, width, height, "head")
                x, y, width, height = right_shoulder_rectangle
                img = deseneaza_dreptunghi(img, x, y, width, height, "right")
                img = deseneaza_linii(img, right_shoulder_lines, color="BLUE")
                img = deseneaza_linii(img, right_shoulder_line, color="RED")

                x, y, width, height = left_shoulder_rectangle
                img = deseneaza_dreptunghi(img, x, y, width, height, "left")
                img = deseneaza_linii(img, left_shoulder_lines, color="BLUE")
                img = deseneaza_linii(img, left_shoulder_line, color="RED")

                eye_left_x = 0
                eye_right_x = 0
                shoulder_left_x = 0
                shoulder_right_x = 0
                if len(eyes) == 2:
                    k = 0
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), 2)

                        if k == 0:
                            eye_left_x = ex
                            k = k + 1
                        if k == 1:
                            eye_right_x = ex

                if eye_left_x > eye_right_x:
                    r = eye_right_x
                    eye_right_x = eye_left_x
                    eye_left_x = r
                if key_presed and len(eyes) == 2 and pos_saved == False :
                    print("eye_left_x = " + str(eye_left_x))
                    print("eye_right_x = " + str(eye_right_x))
                    d_l_es, d_r_es = calculeaza_info_postura(eye_left_x, eye_right_x, left_shoulder_line, right_shoulder_line)
                    pos_saved = True
                    print("Fata salvata, l = " + str(d_l_es) + " r = " + str(d_r_es))

                if pos_saved and len(eyes) == 2:
                    dt_l_es, dt_r_es = calculeaza_info_postura(eye_left_x, eye_right_x, left_shoulder_line, right_shoulder_line)
                    #print("Fata noua l = " + str(dt_l_es) + " r = " + str(dt_r_es))
                    toleranta = 15
                    if dt_l_es < d_l_es - toleranta or dt_r_es < d_r_es - toleranta or dt_l_es > d_l_es + toleranta or dt_r_es > d_r_es + toleranta:
                        if var_normalizare > 3:
                            print("PROBLEM  Stai Drept")
                            winsound.Beep(440, 500)
                            print("Fata noua l = " + str(dt_l_es) + " r = " + str(dt_r_es))
                            var_normalizare = 0
                        else:
                            var_normalizare = var_normalizare + 1

        except Exception as e:
            print(e);
            print('Nuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu')
            print(traceback.format_exc())
            #print("face not found!");
            img = img;

        # draw the face
        cv2.imshow('webcam_feed', img)
    print("Fata salvata finis, l = " + str(d_l_es) + " r = " + str(d_r_es))
        # record frame
        # print(img.shape)
        # img = np.random.randint(255, size=img.shape).astype('uint8')


        # wait so that image has time to draw
        #cv2.waitKey(50) # wait for up to N milliseconds


#################
## init globals
#################

# define rectangle colro
RECTANGLE_COLOR = (0,165,255)

#Open the first video device
capture = cv2.VideoCapture(0)
frame_width = int( capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int( capture.get( cv2.CAP_PROP_FRAME_HEIGHT))
print(frame_height, frame_width);
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
#face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")
#eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
find_and_display(capture);
