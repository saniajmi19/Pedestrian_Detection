import cv2
import numpy as np

def detection(frame1,frame2):
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) < 900:
                continue
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame1, "Pedestrian Detection", (10, 20), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 3)
    return frame1

def segmentation(frame2,fgbg):
    kernel = np.ones((4,4),np.uint8)
    seg = np.zeros(frame2.shape, np.uint8)
    fgmask = fgbg.apply(frame2)
    fgmask = cv2.morphologyEx(fgmask , cv2.MORPH_OPEN, kernel)
    seg[:,:,0] = fgmask
    seg[:,:,1] = fgmask
    seg[:,:,2] = fgmask
    res = cv2.bitwise_and(frame2, frame2, mask=fgmask)
    cv2.putText(seg, "Pedestrian Segmentation", (10, 20), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 3)
    cv2.putText(res, "Background Reduction", (10, 20), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 3)
    return seg,res

def Pedestrian_detection(cap,out):

    fgbg = cv2.createBackgroundSubtractorMOG2()

    while cap.isOpened():
        ret, frame1 = cap.read()
        ret, frame2 = cap.read()
        if ret==True:
            seg,res = segmentation(frame2,fgbg)
            frame1 = detection(frame1,frame2)
            frame_org=frame2.copy()
            cv2.putText(frame_org, "Video streem", (10, 20), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 3)
            img_list1=[frame_org,frame1]
            img_list2=[seg,res]
            img1=np.hstack(img_list1)
            img2=np.hstack(img_list2)
            img_list=[img1,img2]
            img_f=np.vstack(img_list)
            img_f1 = cv2.resize(img_f, (int(img_f.shape[1]//1.5),int(img_f.shape[0]//1.5)))

            cv2.imshow("feed", img_f1)
            out.write(img_f)
            if cv2.waitKey(40) == 27:
                break
        else:
            break

    cv2.destroyAllWindows()
    cap.release()
    out.release()
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--video', metavar='N', type=str, help='give vidoe path')
args = parser.parse_args()
# print((args.video))
if __name__=="__main__":
    cap = cv2.VideoCapture('./input_video/ped1.mp4')
    #cap = cv2.VideoCapture(args.video)
    frame_width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height =int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
    out = cv2.VideoWriter("./output_video/output.mp4", fourcc, 5.0, (int((frame_width*2)),int((frame_height*2))))
    Pedestrian_detection(cap,out)

