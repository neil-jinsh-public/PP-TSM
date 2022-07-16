import cv2

def captureVideoFromCamera():
    url = r'rtsp://demo.easynvr.com:5548/live/stream_17?token=kJ5xszTo'
    cap = cv2.VideoCapture(url, cv2.CAP_DSHOW)
    # WIDTH/HEIGHT必须和摄像头逐帧捕获的分辨率一致，否则会生成1kb视频文件并且无法播放,by Navy 2022-03-31
    # 通过frame.shape获取摄像头逐帧分辨率,by Navy 2022-03-31
    WIDTH = 640
    HEIGHT = 480
    FILENAME = r'Now'

    FPS = 24
    cap.set(cv2.CAP_PROP_FPS, 24)
    # 如下fourcc参数必须是小写，用大写会有OpenCV报错,by Navy 2022-03-31
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(FILENAME, fourcc=fourcc, fps=FPS, frameSize=(WIDTH, HEIGHT))

    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # 逐帧捕获
        ret, frame = cap.read()
        # 如果正确读取帧，ret为True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # 如下通过frame.shape获取摄像头逐帧分辨率,by Navy 2022-03-31
        print(frame.shape)
        # frame = cv2.flip(frame, 1)  # 水平翻转
        ret = out.write(frame)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # gray = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        # 显示结果帧e
        cv2.namedWindow('frame', cv2.WND_PROP_FULLSCREEN)  # 支持全屏,by Navy,2022.04.01
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):  break
    # 完成所有操作后，释放捕获器
    out.release()
    cap.release()
    cv2.destroyAllWindows()


captureVideoFromCamera()
