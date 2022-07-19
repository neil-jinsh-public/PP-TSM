import os
import cv2
import time


def video_save():
    # call video_stream
    camera_id = 0
    cap = cv2.VideoCapture(camera_id)

    # get camera width/height
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # new VideoWriter object
    # para: DIVX，XVID，MJPG，X264，WMV1，WMV2
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # save_path
    path = os.getcwd()
    vpath = path + '\Video'

    if not (os.path.exists(vpath)):
        os.makedirs(vpath)

    fps = 30
    output = cv2.VideoWriter(vpath + '\\' + time.strftime(r"%Y-%m-%d_%H-%M-%S", time.localtime()) + '.avi', fourcc,
                             24.0, (int(width), int(height)))
    frame_Num = 5 * fps

    # read video and save
    while cap.isOpened() and frame_Num > 0:
        ret, frame = cap.read()
        if ret:
            output.write(frame)
            # show video
            # cv2.imshow("window_video", frame)
        # key = cv2.waitKey(1)
        frame_Num -= 1

    # close_video
    cap.release()
    output.release()
    cv2.destroyAllWindows()

    print("Video acquired!")


def main():
    video_save()


if __name__ == '__main__':
    main()
