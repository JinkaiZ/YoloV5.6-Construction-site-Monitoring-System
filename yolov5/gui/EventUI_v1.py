# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'EventUI_v1.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(514, 582)
        self.verticalLayoutWidget = QtWidgets.QWidget(Form)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(20, 20, 471, 531))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_title = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_title.setObjectName("label_title")
        self.horizontalLayout_2.addWidget(self.label_title)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_imagePlaceHolder = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_imagePlaceHolder.setObjectName("label_imagePlaceHolder")
        self.horizontalLayout.addWidget(self.label_imagePlaceHolder)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_description = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_description.setObjectName("label_description")
        self.verticalLayout_2.addWidget(self.label_description)
        self.label_eventType = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_eventType.setObjectName("label_eventType")
        self.verticalLayout_2.addWidget(self.label_eventType)
        self.label_time = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_time.setObjectName("label_time")
        self.verticalLayout_2.addWidget(self.label_time)
        self.label_date = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_date.setObjectName("label_date")
        self.verticalLayout_2.addWidget(self.label_date)
        self.pushButton_report = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_report.setObjectName("pushButton_report")
        self.verticalLayout_2.addWidget(self.pushButton_report)
        self.pushButton_delete = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_delete.setObjectName("pushButton_delete")
        self.verticalLayout_2.addWidget(self.pushButton_delete)
        self.verticalLayout.addLayout(self.verticalLayout_2)
        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(1, 6)
        self.verticalLayout.setStretch(2, 3)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label_title.setText(_translate("Form", "Event - 1"))
        self.label_imagePlaceHolder.setText(_translate("Form", "Image"))
        self.label_description.setText(_translate("Form", "Description: "))
        self.label_eventType.setText(_translate("Form", "Event Type"))
        self.label_time.setText(_translate("Form", "Time"))
        self.label_date.setText(_translate("Form", "Date"))
        self.pushButton_report.setText(_translate("Form", "Report"))
        self.pushButton_delete.setText(_translate("Form", "Delete"))