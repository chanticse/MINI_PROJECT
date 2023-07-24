import numpy as np
import cv2
import sys
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.PlotModule import LivePlot
import cvzone
import time
from keras.models import load_model
import os
from pygame import mixer

real_Width = 640   
real_Height = 480  
video_Channels = 3
video_FrameRate = 15
video_Width = 160
video_Height = 120

mixer.init()
alarm_sound = mixer.Sound('alarm.wav')



# Drowsiness Detection Parameters
def detect_drowsiness(frame):
    
    facedetection = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
    left_eyedetection = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
    right_eyedetection = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')

    labels = ['Close', 'Open']
    model = load_model('models/custmodel.h5')

   
    path = os.getcwd()

    font = cv2.FONT_HERSHEY_COMPLEX_SMALL

    counter = 0
    time_count = 0
    thick = 2
    right_eye_pred = [99]
    left_eye_pred = [99]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetection.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = left_eyedetection.detectMultiScale(gray)
    right_eye = right_eyedetection.detectMultiScale(gray)


    cv2.rectangle(frame, (0, real_Height - 50), (100, real_Height), (0, 0, 0), thickness=cv2.FILLED)
    cv2.rectangle(frame, (290, real_Height - 50), (540, real_Height), (0, 0, 0), thickness=cv2.FILLED)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

    for (x, y, w, h) in right_eye:
        right = frame[y:y + h, x:x + w]
        counter += 1
        right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        right = cv2.resize(right, (24, 24))
        right = right / 255
        right = right.reshape(24, 24, -1)
        right = np.expand_dims(right, axis=0)
        right_eye_pred = np.argmax(model.predict(right), axis=-1)

        if right_eye_pred[0] == 1:
            labels = 'Open'
        if right_eye_pred[0] == 0:
            labels = 'Closed'
        break

    for (x, y, w, h) in left_eye:
        left = frame[y:y + h, x:x + w]
        counter += 1
        left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        left = cv2.resize(left, (24, 24))
        left = left / 255
        left = left.reshape(24, 24, -1)
        left = np.expand_dims(left, axis=0)
        left_eye_pred = np.argmax(model.predict(left), axis=-1)

        if left_eye_pred[0] == 1:
            labels = 'Open'
        if left_eye_pred[0] == 0:
            labels = 'Closed'
        break

    if right_eye_pred[0] == 0 and left_eye_pred[0] == 0:
        time_count += 1
        cv2.putText(frame, "Inactive", (10, real_Height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        time_count -= 1
        cv2.putText(frame, "Active", (10, real_Height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if time_count < 0:
        time_count = 0

    cv2.putText(frame, 'Wake up Time !!:' + str(time_count), (300, real_Height - 20), font, 1, (0, 0, 255), 1,
                cv2.LINE_AA)

    if time_count > 10:
        cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
        try:
            alarm_sound.play()
        except:
            pass

        if thick < 16:
            thick = thick + 2
        else:
            thick = thick - 2
            if thick < 2:
                thick = 2

        cv2.rectangle(frame, (0, 0), (real_Width, real_Height), (0, 0, 255), thick)

    return frame
def calculate_heartrate(webcam):

    detector = FaceDetector()

    webcam.set(3, real_Width)
    webcam.set(4, real_Height)

    # Color Magnification Parameters
    levels = 3
    alpha = 170
    min_Frequency = 1.0
    max_Frequency = 2.0
    buffer_Size = 150
    bufferIndex = 0

    plotY = LivePlot(real_Width, real_Height, [60, 120], invert=True)

    # Helper Methods
    def build_Gauss(frame, levels):
        pyramid = [frame]
        for level in range(levels):
            frame = cv2.pyrDown(frame)
            pyramid.append(frame)
        return pyramid

    def reconstruct_Frame(pyramid, index, levels):
        filteredFrame = pyramid[index]
        for level in range(levels):
            filteredFrame = cv2.pyrUp(filteredFrame)
        filteredFrame = filteredFrame[:video_Height, :video_Width]
        return filteredFrame


    # Output Display Parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    loadingTextLocation = (30, 40)
    bpmTextLocation = (video_Width // 2, 40)


    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2
    boxColor = (0, 255, 0)
    boxWeight = 3

    # Initialize Gaussian Pyramid
    firstFrame = np.zeros((video_Height, video_Width, video_Channels))
    firstGauss = build_Gauss(firstFrame, levels+1)[levels]
    videoGauss = np.zeros((buffer_Size, firstGauss.shape[0], firstGauss.shape[1], video_Channels))
    fourierTransformAvg = np.zeros((buffer_Size))

    # Bandpass Filter for Specified Frequencies
    frequencies = (1.0 * video_FrameRate) * np.arange(buffer_Size) / (1.0 * buffer_Size)
    mask = (frequencies >= min_Frequency) & (frequencies <= max_Frequency)

    # Heart Rate Calculation Variables
    bpmCalculationFrequency = 10   #15
    bpmBufferIndex = 0
    bpmBufferSize = 10
    bpmBuffer = np.zeros((bpmBufferSize))

    i=0
    ptime = 0
    ftime = 0

    while True:
        ret, frame = webcam.read()
        if ret == False:
            break

        frame, bboxs = detector.findFaces(frame, draw=False)
        frameDraw = frame.copy()
        ftime = time.time()
        fps = 1 / (ftime - ptime)
        ptime = ftime


        if bboxs:
            x1, y1, w1, h1 = bboxs[0]['bbox']
            cv2.rectangle(frameDraw, bboxs[0]['bbox'], (255, 0, 255), 2)
            detectionFrame = frame[y1:y1 + h1, x1:x1 + w1]
            detectionFrame = cv2.resize(detectionFrame, (video_Width, video_Height))

            # Construct Gaussian Pyramid
            videoGauss[bufferIndex] = build_Gauss(detectionFrame, levels+1)[levels]
            fourierTransform = np.fft.fft(videoGauss, axis=0)

            # Bandpass Filter
            fourierTransform[mask == False] = 0

            # Grab a Pulse
            if bufferIndex % bpmCalculationFrequency == 0:
                i = i + 1
                for buf in range(buffer_Size):
                    fourierTransformAvg[buf] = np.real(fourierTransform[buf]).mean()
                hz = frequencies[np.argmax(fourierTransformAvg)]
                bpm=60.0*hz
                bpmBuffer[bpmBufferIndex] = bpm
                bpmBufferIndex = (bpmBufferIndex + 1) % bpmBufferSize

            # Amplify
            filtered = np.real(np.fft.ifft(fourierTransform, axis=0))
            filtered = filtered * alpha

            # Reconstruct Resulting Frame
            filteredFrame = reconstruct_Frame(filtered, bufferIndex, levels)
            outputFrame = detectionFrame + filteredFrame
            outputFrame = cv2.convertScaleAbs(outputFrame)

            bufferIndex = (bufferIndex + 1) % buffer_Size
            outputFrame_show = cv2.resize(outputFrame, (video_Width // 2, video_Height // 2))
            frameDraw[0:video_Height // 2, (real_Width - video_Width // 2):real_Width] = outputFrame_show

            bpm_value = int(bpmBuffer.mean())
            imgPlot = plotY.update(int(bpm_value))

            if i > bpmBufferSize:
                cvzone.putTextRect(frameDraw, f'BPM: {bpm_value}', bpmTextLocation, scale=2)
            else:
                cvzone.putTextRect(frameDraw, "Calculating BPM...", loadingTextLocation, scale=2)
            
            frame_with_drowsiness = detect_drowsiness(frameDraw)

            imgStack = cvzone.stackImages([frame_with_drowsiness, imgPlot], 2, 1)
            cv2.imshow("Heart Rate and Drowsiness Detection", imgStack)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            frame_with_drowsiness = detect_drowsiness(frameDraw)
            imgStack = cvzone.stackImages([frame_with_drowsiness, frame_with_drowsiness], 2, 1)
            cv2.imshow("Heart Rate and Drowsiness Detection", imgStack)

    webcam.release()
    cv2.destroyAllWindows()

def main():
    webcam = cv2.VideoCapture(0)
    calculate_heartrate(webcam)

if __name__ == '__main__':
    main()
