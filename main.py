import cv2
import numpy as np

def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        if mirror:
            img = cv2.flip(img, 1)
        cv2.imshow('my webcam', img)
        output = img.copy()
        image = img.copy()


        boundaries = [

            ([100, 100, 100], [255, 255, 255])
        ]

        # loop over the boundaries

        #for (lower, upper) in boundaries:
        #    # create NumPy arrays from the boundaries
        #    lower = np.array(lower, dtype="uint8")
        #    upper = np.array(upper, dtype="uint8")
             # the colors within the specified boundaries and apply
             #the mask
        #    mask = cv2.inRange(image, lower, upper)
        #    output = cv2.bitwise_and(image, image, mask=mask)
        #    cv2.imshow("color", output)

        gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.4, 200, maxRadius=75)
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for(x,y,r) in circles:
                cv2.circle(output, (x,y), r, (0,255,0),4)
                print(len(circles))

        cv2.imshow("circles", output)

        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()
    #change to fjdklsajfdl;ksajfl;dsa


def main():
    show_webcam(mirror=True)


if __name__ == '__main__':
    main()