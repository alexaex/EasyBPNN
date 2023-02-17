CMAKE_MINIMUM_REQUIRED(VERSION 3.24)
SET(CMAKE_CXX_STANDARD 14)

PROJECT(MLP)
# MULTIPLE LAYER PERCEPTRON


INCLUDE_DIRECTORIES(
        include/Eigen
        include/DNN
)

ADD_EXECUTABLE(
        MLP
        src/main.cpp
        src/module.cpp src/optimizer_utils.cpp)
