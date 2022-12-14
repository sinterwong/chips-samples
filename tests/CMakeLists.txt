CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
PROJECT(Tests)

INCLUDE_DIRECTORIES(
    ${PROJECT_INCLUDE_DIR}
    ${DNN_PATH}/include
    ${APPSDK_PATH}/include
    ${APPSDK_PATH}/include/vio
    ${APPSDK_PATH}/include/libmm
)

LINK_DIRECTORIES(
    ${DNN_LIB_PATH}
    ${APPSDK_PATH}/lib
    ${APPSDK_PATH}/lib/hbbpu
    ${APPSDK_PATH}/lib/hbmedia
)

LINK_LIBRARIES(
    gflags::gflags
    pthread
    module_logger
    ${BPU_libs}
    ${HB_MEDIA_libs}
    rt
    dl
)

FILE(GLOB APP_SOURCES *.cpp)
MESSAGE(STATUS "APP SOURCES: ${APP_SOURCES}")
FOREACH(sourcefile ${APP_SOURCES})
    STRING(REGEX MATCH "[^/]+$" sourcefilewithoutpath ${sourcefile})
    STRING(REPLACE ".cpp" "" testname ${sourcefilewithoutpath})
    ADD_EXECUTABLE( ${testname} ${sourcefile})
    SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin/tests)
    
    INSTALL(TARGETS ${testname}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)
ENDFOREACH(sourcefile ${APP_SOURCES})
