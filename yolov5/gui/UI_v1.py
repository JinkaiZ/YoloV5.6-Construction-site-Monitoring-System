# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UI_v1.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1270, 737)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(20, 20, 1231, 641))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(6)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.horizontalLayout_mainWindow = QtWidgets.QHBoxLayout()
        self.horizontalLayout_mainWindow.setObjectName("horizontalLayout_mainWindow")
        self.verticalLayout_videoSection = QtWidgets.QVBoxLayout()
        self.verticalLayout_videoSection.setObjectName("verticalLayout_videoSection")
        self.label_Title = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.label_Title.setTextFormat(QtCore.Qt.AutoText)
        self.label_Title.setObjectName("label_Title")
        self.verticalLayout_videoSection.addWidget(self.label_Title)
        self.label_videoPlaceHolder = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.label_videoPlaceHolder.setObjectName("label_videoPlaceHolder")
        self.verticalLayout_videoSection.addWidget(self.label_videoPlaceHolder)
        self.gridLayout_buttonSection = QtWidgets.QGridLayout()
        self.gridLayout_buttonSection.setObjectName("gridLayout_buttonSection")
        self.pushButton_4 = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pushButton_4.setObjectName("pushButton_4")
        self.gridLayout_buttonSection.addWidget(self.pushButton_4, 1, 1, 1, 1)
        self.pushButton_2 = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.gridLayout_buttonSection.addWidget(self.pushButton_2, 1, 0, 1, 1)
        self.pushButton_start = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pushButton_start.setObjectName("pushButton_start")
        self.gridLayout_buttonSection.addWidget(self.pushButton_start, 0, 0, 1, 1)
        self.pushButton_stop = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pushButton_stop.setObjectName("pushButton_stop")
        self.gridLayout_buttonSection.addWidget(self.pushButton_stop, 0, 1, 1, 1)
        self.pushButton_3 = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pushButton_3.setObjectName("pushButton_3")
        self.gridLayout_buttonSection.addWidget(self.pushButton_3, 2, 0, 1, 1)
        self.pushButton = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pushButton.setObjectName("pushButton")
        self.gridLayout_buttonSection.addWidget(self.pushButton, 2, 1, 1, 1)
        self.verticalLayout_videoSection.addLayout(self.gridLayout_buttonSection)
        self.verticalLayout_videoSection.setStretch(0, 1)
        self.verticalLayout_videoSection.setStretch(1, 6)
        self.verticalLayout_videoSection.setStretch(2, 3)
        self.horizontalLayout_mainWindow.addLayout(self.verticalLayout_videoSection)
        self.verticalLayout_riskEventSection = QtWidgets.QVBoxLayout()
        self.verticalLayout_riskEventSection.setObjectName("verticalLayout_riskEventSection")
        self.label_RiskEvent = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.label_RiskEvent.setObjectName("label_RiskEvent")
        self.verticalLayout_riskEventSection.addWidget(self.label_RiskEvent)
        self.tableWidget_eventDisplaySection = QtWidgets.QTableWidget(self.horizontalLayoutWidget)
        self.tableWidget_eventDisplaySection.setObjectName("tableWidget_eventDisplaySection")
        self.tableWidget_eventDisplaySection.setColumnCount(4)
        self.tableWidget_eventDisplaySection.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_eventDisplaySection.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_eventDisplaySection.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_eventDisplaySection.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_eventDisplaySection.setHorizontalHeaderItem(3, item)
        self.verticalLayout_riskEventSection.addWidget(self.tableWidget_eventDisplaySection)
        self.horizontalLayout_mainWindow.addLayout(self.verticalLayout_riskEventSection)
        self.horizontalLayout_mainWindow.setStretch(0, 8)
        self.horizontalLayout_mainWindow.setStretch(1, 4)
        self.horizontalLayout.addLayout(self.horizontalLayout_mainWindow)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1270, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_Title.setText(_translate("MainWindow", "Construction site Monitoring System"))
        self.label_videoPlaceHolder.setText(_translate("MainWindow", "Video"))
        self.pushButton_4.setText(_translate("MainWindow", "Steam"))
        self.pushButton_2.setText(_translate("MainWindow", "Camera"))
        self.pushButton_start.setText(_translate("MainWindow", "Start"))
        self.pushButton_stop.setText(_translate("MainWindow", "Stop"))
        self.pushButton_3.setText(_translate("MainWindow", "Select Model"))
        self.pushButton.setText(_translate("MainWindow", "Train new model"))
        self.label_RiskEvent.setText(_translate("MainWindow", "Risk Event"))
        item = self.tableWidget_eventDisplaySection.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Picture"))
        item = self.tableWidget_eventDisplaySection.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Event"))
        item = self.tableWidget_eventDisplaySection.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "Time"))
        item = self.tableWidget_eventDisplaySection.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "Date"))
