QT += core
QT -= gui

CONFIG += c++11

TARGET = slam-ch3
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += \
    ../../eigenGeometry.cpp
INCLUDEPATH += /usr/include \
               /usr/local/include \
                /usr/local/include/eigen3

