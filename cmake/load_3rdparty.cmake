# Once done, this will define
#  ......

SET(3RDPARTY_ROOT ${PROJECT_SOURCE_DIR}/3rdparty)
SET(3RDPARTY_DIR ${PROJECT_SOURCE_DIR}/3rdparty/target/${TARGET_OS}_${TARGET_ARCH}_${TARGET_HARDWARE})
MESSAGE(STATUS "3RDPARTY_DIR: ${3RDPARTY_DIR}")

MACRO(LOAD_X3)
   # define dnn lib path
    SET(DNN_PATH "~/.horizon/ddk/xj3_aarch64/dnn/")
    SET(APPSDK_PATH "~/.horizon/ddk/xj3_aarch64/appsdk/appuser/")

    SET(DNN_LIB_PATH ${DNN_PATH}/lib)
    SET(BPU_libs dnn cnn_intf hbrt_bernoulli_aarch64)
    SET(HB_MEDIA_libs vio hbmedia avcodec avformat avutil)

ENDMACRO()
