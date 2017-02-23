QT += core
QT -= gui

CONFIG += c++11

TARGET = slam-ch5
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += \
    ../../pointCloud.cpp \
    ../../pcl_show.cpp
INCLUDEPATH += /usr/include \
               /usr/local/include \
                /usr/local/include/eigen3 \
                /usr/include/pcl-1.7 \
