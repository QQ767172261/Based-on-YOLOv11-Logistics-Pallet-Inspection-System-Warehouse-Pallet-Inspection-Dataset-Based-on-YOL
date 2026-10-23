仓库仓储托盘检测数据集 基于 YOLOv11 物流托盘检测系统 仓库仓储托盘检测数据集 基于 YOLOv11 物流托盘检测系统![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/dca7f3e715b74d1a9bc50baf01a01bd2.png)
，4726张，yolo，voc，coco三种标注方式
图像尺寸:640*640
类别数量:1类![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5204be60664f40e2915b40629e915c81.png)

训练集图像数量:3324; 验证集图像数量:938； 测试集图像数量:464
类别名称: 每一类图像数 ，每一类标注数
pallet-托盘: 4726,120418
image num: 4726![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/69e0cd92ee9e4c098d5f1029efa22ec9.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bb1ba501852d4bb383b40bbf30968423.png)



模型代码：模型训练使用yolov11n训练，50个epoch训练结果，map如描述图所示。![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/cd4e4b5fff7e45c3a47451488f739757.png)

qt界面：运行界面采用pyqt编写，本项目已经训练好模型，配置好环境后可直接使用，运行效果见描述图像![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7b2ac06f30b44537a37f7be44e3f54ea.png)



4.（支持windows和linux系统）

# 基于YOLOv11 物流托盘检测系统完整代码
## 1、环境安装
```bash
pip install ultralytics torch opencv-python pyqt5 numpy
```

## 2、数据集配置文件 pallet.yaml
```yaml
path: ./pallet_dataset
train: images/train
val: images/val
test: images/test

names:
  0: pallet
```

## 3、模型训练脚本 train_pallet.py
```python
from ultralytics import YOLO

if __name__ == '__main__':
    # 加载yolo11n预训练权重
    model = YOLO("yolo11n.pt")
    # 开始训练
    train_result = model.train(
        data="pallet.yaml",
        epochs=80,
        imgsz=640,
        batch=12,
        device=0,
        project="pallet_result",
        name="yolo11_pallet",
        conf=0.25,
        iou=0.45
    )
```

## 4、PyQt5可视化系统完整代码 pallet_gui.py
```python
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QFileDialog,
                             QLabel, QTableWidget, QTableWidgetItem, QComboBox, QSpinBox)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from ultralytics import YOLO

class PalletDetectWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于YOLOv11的物流托盘检测系统")
        self.setGeometry(100,100,1200,800)
        self.model = YOLO("./pallet_result/yolo11_pallet/weights/best.pt")
        self.conf_thresh = 0.25

        # 控件
        self.img_label = QLabel(self)
        self.img_label.setGeometry(20,80,700,550)
        self.img_label.setStyleSheet("border:1px solid #000;")

        self.btn_img = QPushButton("导入图片",self)
        self.btn_img.setGeometry(750,80,180,40)
        self.btn_img.clicked.connect(self.load_image)

        self.btn_video = QPushButton("导入视频",self)
        self.btn_video.setGeometry(750,130,180,40)

        self.label_conf = QLabel("置信度",self)
        self.label_conf.setGeometry(750,180,80,30)
        self.spin_conf = QSpinBox(self)
        self.spin_conf.setGeometry(820,180,80,30)
        self.spin_conf.setValue(25)
        self.spin_conf.valueChanged.connect(self.change_conf)

        self.table = QTableWidget(self)
        self.table.setGeometry(20,650,700,120)
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["序号","文件路径","类别","置信度","xmin,ymin","xmax,ymax"])

    def change_conf(self,val):
        self.conf_thresh = val / 100

    def load_image(self):
        path,_ = QFileDialog.getOpenFileName(self,"选择图片","", "Image(*.jpg *.png *.jpeg)")
        if not path:
            return
        img = cv2.imread(path)
        results = self.model(img, conf=self.conf_thresh)
        self.table.setRowCount(0)
        idx = 1
        for res in results:
            boxes = res.boxes
            for box in boxes:
                x1,y1,x2,y2 = map(int,box.xyxy[0])
                conf = float(box.conf)
                cls_name = self.model.names[int(box.cls)]
                cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
                cv2.putText(img,f"{cls_name} {conf:.2f}",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
                # 写入表格
                row = self.table.rowCount()
                self.table.insertRow(row)
                self.table.setItem(row,0,QTableWidgetItem(str(idx)))
                self.table.setItem(row,1,QTableWidgetItem(path))
                self.table.setItem(row,2,QTableWidgetItem(cls_name))
                self.table.setItem(row,3,QTableWidgetItem(f"{conf:.3f}"))
                self.table.setItem(row,4,QTableWidgetItem(f"{x1},{y1}"))
                self.table.setItem(row,5,QTableWidgetItem(f"{x2},{y2}"))
                idx +=1
        # 图片转qt显示
        img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        h,w,c = img_rgb.shape
        qimg = QImage(img_rgb.data,w,h,w*c,QImage.Format_RGB888)
        self.img_label.setPixmap(QPixmap.fromImage(qimg).scaled(self.img_label.size(),Qt.KeepAspectRatio))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = PalletDetectWindow()
    win.show()
    sys.exit(app.exec_())
```

## 5、应用场景
1. 智能仓储货架区域托盘数量自动统计，盘点库存；
2. 仓库安防摄像头实时检测空托盘/载物托盘；
3. 物流园区AGV叉车视觉辅助识别托盘位置；
4. 仓储数字化管理，自动识别托盘堆放超限、占道。

## 标签
`pallet`：物流托盘


