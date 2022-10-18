import glob
import shutil
import time
from datetime import datetime
from PIL import Image
from natsort import natsorted
import numpy as np
from PIL.Image import Image
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu, QAction, QTableWidgetItem
from gui.UI_v1 import Ui_MainWindow
from PyQt5.QtCore import Qt, QPoint, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QIcon

import pymysql
import base64
import constants

import argparse
import os
import sys
from pathlib import Path
from natsort import natsorted
import torch
import torch.backends.cudnn as cudnn

from yolov5.gui.EventUI_v1 import Ui_Form

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

class TrainThread(QThread):
    def __init__(self, parent=None):
        super(TrainThread, self).__init__(parent)
    def run(self):
        os.system(
            'cd ../yolov5 && python train.py --img 640 --batch 16 --epochs 10 --data dataset.yaml --weights yolov5s.pt --workers=2')






class DetThread(QThread):

    send_img = pyqtSignal(np.ndarray)
    update_data = pyqtSignal()
    jump_out = False
    currentWeight = '../yolov5/Trained models/Safety Vest.pt'

    def __int__(self):
        super(DetThread, self).__int__()
        self.running = True   # break the while loop if True
        self.source = '0'            # default input source is webcam


    @torch.no_grad()
    def run(self,
            running=True,  # break the while loop if True
            riskFlag = False,
            weights='../yolov5/Trained models/Safety Hardhat.pt',  # model.pt path(s)
            data=ROOT / 'dataset.yaml',  # dataset.yaml path
            imgsz=(640, 640),  # inference size (height, width)
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=False,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project=ROOT / 'riskEvents',  # save risk events to project/name
            name='0',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
            ):



        source = str(self.source)
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
        if is_url and is_file:
            source = check_file(source)  # download

        # Directories - save to runs/detect/exp*
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment risk folder risk2...
        # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)  # data = dataset.yaml
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Dataloader
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
            bs = len(dataset)  # batch_size
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
            bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
        dt, seen = [0.0, 0.0, 0.0], 0

        # count the risk label
        count = 0


        dataset = iter(dataset)

        while True:

            if self.jump_out:
                dataset = None
                break

            if self.currentWeight != weights:
                weights = self.currentWeight
                device = select_device(device)
                model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)  # data = dataset.yaml
                stride, names, pt = model.stride, model.names, model.pt
                imgsz = check_img_size(imgsz, s=stride)  # check image size

            path, im, im0s, vid_cap, s = next(dataset)
            # print(self.check)
            t1 = time_sync()
            im = torch.from_numpy(im).to(device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                # save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + (
                    '' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string


                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))


                # Stream results & send the rendered video to Qt slot
                im0 = annotator.result()
                if view_img:
                    # cv2.imshow(str(p), im0)
                    self.send_img.emit(im0)
                    cv2.waitKey(1)  # 1 millisecond


                # Determine the risk situation exists more than 3 seconds or not
                # Use an array to hold all the risk labels later
                if 'non hardhat wearing' or 'non safety glass wearing' or 'non safety vest wearing ' in s:
                    count = count + 1
                else:
                    count = 0


                if count > 100:
                    riskFlag = True
                    # write the date & time into a text file
                    if 'non hardhat wearing' in s:
                        risk = 'non hardhat wearing,'
                    if 'non safety glass wearing' in s:
                        risk = 'non safety glass wearing,'
                    if 'non safety vest wearing' in s:
                        risk = 'non safety vest wearing,'
                    count = 0

                # Save risk results (image with detections)
                if riskFlag:
                    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment risk folder risk2...
                    # create the folder
                    os.mkdir(save_dir)
                    print(save_dir)
                    path = str(save_dir / 'picture.jpg')  # im.jpg

                    # save the risk picture
                    cv2.imwrite(path, im0)

                    # Get the current date & time
                    now = datetime.now()
                    current_date = now.strftime("%Y:%m:%d")
                    current_time = now.strftime("%H:%M:%S")



                    with open(save_dir / 'riskInfo.txt', 'w') as f:
                        f.write(risk + current_date + ',' + current_time)


                    riskFlag = False
                    count = 0
                    self.update_data.emit()

                    # Upload risk result to SQL
                    try:
                        # Connect DB
                        conn = pymysql.connect(
                            host=constants.HOST, user=constants.USER, password=constants.PASSWORD,
                            db=constants.TRAINING_DATA, charset=constants.ENCODING,
                        )
                        print("Connect to DB successfully！")

                        # Use cursor that returns result in diciotnary format
                        cursor = conn.cursor()

                        # Read picture.jpg as binary data

                        with open(path, 'rb') as f:
                            image = f.read()
                            image = base64.b64encode(image)

                        riskTitle = risk.rstrip(risk[-1])
                        print(riskTitle)
                        title = riskTitle
                        date = current_date
                        time = current_time
                        date_time_str = date + " " + time
                        date_time_obj = datetime.strptime(date_time_str, '%Y:%m:%d %H:%M:%S')
                        print(date_time_obj)

                        cursor.execute(
                        '''
                        INSERT INTO hazard (TITLE, DESCRIPTION, DATE, IMAGE)
                        VALUES (%s, %s, %s, %s)
                        ''', (title, "The worker has " + riskTitle, date_time_obj, image))
                        print("insert successfully")
                        conn.commit()
                        print("commit successfully")

                        cursor.close()
                    except Exception as e:
                        print(e)



            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')




class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)

        self.det_thread = DetThread()
        self.det_thread.update_data.connect(self.load_risk_events_data_table)

        self.tableWidget_eventDisplaySection.setColumnWidth(0,70)
        self.tableWidget_eventDisplaySection.setColumnWidth(1, 120)
        self.tableWidget_eventDisplaySection.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tableWidget_eventDisplaySection.doubleClicked.connect(self.event_detail)

        self.pushButton_start.clicked.connect(self.start_detection)
        self.pushButton_stop.clicked.connect(self.stop_detection)
        self.pushButton_clearALL.clicked.connect(self.clear_all)
        self.pushButton_trainModel.clicked.connect(self.train_model)
        self.det_thread.send_img.connect(lambda x: self.show_image(x, self.label_videoPlaceHolder))
        self.load_risk_events_data_table()

        # auto search the model
        self.modelSelectionComboBox.clear()
        self.pt_list = os.listdir('../yolov5/Trained models')
        self.pt_list = [file for file in self.pt_list if file.endswith('.pt')]
        self.modelSelectionComboBox.clear()
        self.modelSelectionComboBox.addItems(self.pt_list)



    def open_detail_event(self,row):
        self.window = QtWidgets.QWidget()
        self.ui = Ui_Form()
        self.ui.setupUi(self.window)

        # load picture
        if row == 0:
            pic = '../yolov5/riskEvents/0/picture.jpg'
            label_path = '../yolov5/riskEvents/0/riskInfo.txt'
        else:
            row = row
            pic = '../yolov5/riskEvents/0' + str(row) + '/picture.jpg'
            label_path = '../yolov5/riskEvents/0' + str(row) + '/riskInfo.txt'

        self.ui.label_imagePlaceHolder.setScaledContents(True)
        pixmap = QtGui.QPixmap(pic)
        self.ui.label_imagePlaceHolder.setPixmap(pixmap)

        # load event title
        event_no = row + 1
        self.ui.label_title.setText('Risk Event - ' + str(event_no))

        # load descriptions
        with open(label_path) as f:
            line = f.readlines()
            event = line[0].split(',')[0]
            time = line[0].split(',')[2]
            date = line[0].split(',')[1]
            self.ui.label_eventType.setText('The risk events ' + event + ' occurred')
            self.ui.label_time.setText('Time: ' + time)
            self.ui.label_date.setText('Date: ' + date)


        self.window.show()

    def event_detail(self):
        row = self.tableWidget_eventDisplaySection.currentIndex().row()
        self.open_detail_event(row)

    def train_model(self):
        self.train_thread = TrainThread()
        self.train_thread.start()

    def clear_all(self):
        folder = '../yolov5/riskEvents'
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
        self.load_risk_events_data_table()

    def load_risk_events_data_table(self):
        while(self.tableWidget_eventDisplaySection.rowCount() > 0):
            self.tableWidget_eventDisplaySection.removeRow(0)
        # assign directory
        directory = "../yolov5/riskEvents"
        # iterate over files in
        # that directory
        # files = Path(directory).glob('*')
        files = sorted(os.listdir(directory), key=lambda x: (int(x)))
        print(files)

        for file in files:

            i = int(file)

            # load picture
            pic = directory + '/'+ str(file) + '/picture.jpg'
            label = QtWidgets.QLabel()
            label.setText("")
            label.setScaledContents(True)
            pixmap = QtGui.QPixmap(pic)
            label.setPixmap(pixmap)
            self.tableWidget_eventDisplaySection.insertRow(i)
            self.tableWidget_eventDisplaySection.setRowHeight(i,50)
            self.tableWidget_eventDisplaySection.setCellWidget(i,0,label)

            label_path = directory + '/'+ str(file) + '/riskInfo.txt'
            with open(label_path) as f:
                line = f.readlines()
                label = line[0].split(',')[0]
                time = line[0].split(',')[2]
                date = line[0].split(',')[1]
                self.tableWidget_eventDisplaySection.setItem(i,1,QtWidgets.QTableWidgetItem(label))
                self.tableWidget_eventDisplaySection.setItem(i, 2, QtWidgets.QTableWidgetItem(time))
                self.tableWidget_eventDisplaySection.setItem(i, 3, QtWidgets.QTableWidgetItem(date))


    def start_detection(self):
        self.det_thread.jump_out = False
        if not self.det_thread.isRunning():
            self.det_thread.source = '0'
            print(self.modelSelectionComboBox.currentText())
            self.det_thread.currentWeight = '../yolov5/Trained models/' + self.modelSelectionComboBox.currentText()
            self.det_thread.start()



    def stop_detection(self):
        # self.det_thread.terminate()
        # self.det_thread.send_img.connect(lambda x: self.show_image(x, self.label_videoPlaceHolder))
        self.det_thread.jump_out = True



    @staticmethod
    def show_image(img_src, label):
        try:
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()

            # keep the ratio
            if iw > ih:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))

            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))


        except Exception as e:
            print(repr(e))



if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MainWindow()
    myWin.show()
    sys.exit(app.exec_())
