
#include "dnn/hb_dnn.h"
#include "gflags/gflags.h"
#include "hb_comm_vdec.h"
#include "hb_common.h"
#include "hb_type.h"
#include "hb_vdec.h"
#include "hb_vp_api.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <unistd.h>
#include <unordered_map>
#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
#include <libavformat/avformat.h>
#include <libavutil/timestamp.h>
#ifdef __cplusplus
}
#endif /* __cplusplus */

DEFINE_string(video, "", "Specify the video uri.");
DEFINE_string(model_path, "", "Specify the model path.");

#define HB_CHECK_SUCCESS(value, errmsg)                                        \
  do {                                                                         \
    /*value can be call of function*/                                          \
    int ret_code = value;                                                      \
    if (ret_code != 0) {                                                       \
      printf("[BPU ERROR] %s, error code:%d\n", errmsg, ret_code);             \
      return ret_code;                                                         \
    }                                                                          \
  } while (0);

#define SET_BYTE(_p, _b) *_p++ = (unsigned char)_b;

#define SET_BUFFER(_p, _buf, _len)                                             \
  memcpy(_p, _buf, _len);                                                      \
  (_p) += (_len);

struct alignas(float) DetectionRet {
  std::array<float, 4> bbox; // 框
  float confidence;          // 置信度
  float classId;             // 类别
};

static inline bool compare(DetectionRet const &a, DetectionRet const &b) {
  return a.confidence > b.confidence;
}

float iou(std::array<float, 4> const &lbox, std::array<float, 4> const &rbox) {
  float interBox[] = {
      std::max(lbox[0] - lbox[2] / 2.f, rbox[0] - rbox[2] / 2.f), // left
      std::min(lbox[0] + lbox[2] / 2.f, rbox[0] + rbox[2] / 2.f), // right
      std::max(lbox[1] - lbox[3] / 2.f, rbox[1] - rbox[3] / 2.f), // top
      std::min(lbox[1] + lbox[3] / 2.f, rbox[1] + rbox[3] / 2.f), // bottom
  };

  if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
    return 0.0f;

  float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
  return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}

void nms(std::vector<DetectionRet> &res,
         std::unordered_map<int, std::vector<DetectionRet>> &m,
         float nms_thr = 0.45) {
  for (auto it = m.begin(); it != m.end(); it++) {
    // std::cout << it->second[0].class_id << " --- " << std::endl;
    auto &dets = it->second;
    std::sort(dets.begin(), dets.end(), compare);
    for (size_t m = 0; m < dets.size(); ++m) {
      auto &item = dets[m];
      res.push_back(item);
      for (size_t n = m + 1; n < dets.size(); ++n) {
        if (iou(item.bbox, dets[n].bbox) > nms_thr) {
          dets.erase(dets.begin() + n);
          --n;
        }
      }
    }
  }
}

void renderOriginShape(std::vector<DetectionRet> &results,
                       std::array<int, 3> const &shape,
                       std::array<int, 3> const &inputShape,
                       bool isScale = true) {
  float rw, rh;
  if (isScale) {
    rw = std::min(inputShape[0] * 1.0 / shape.at(0),
                  inputShape[1] * 1.0 / shape.at(1));
    rh = rw;
  } else {
    rw = inputShape[0] * 1.0 / shape.at(0);
    rh = inputShape[1] * 1.0 / shape.at(1);
  }

  for (auto &ret : results) {
    int l = (ret.bbox[0] - ret.bbox[2] / 2.f) / rw;
    int t = (ret.bbox[1] - ret.bbox[3] / 2.f) / rh;
    int r = (ret.bbox[0] + ret.bbox[2] / 2.f) / rw;
    int b = (ret.bbox[1] + ret.bbox[3] / 2.f) / rh;
    ret.bbox[0] = l > 0 ? l : 0;
    ret.bbox[1] = t > 0 ? t : 0;
    ret.bbox[2] = r < shape[0] ? r : shape[0];
    ret.bbox[3] = b < shape[1] ? b : shape[1];
  }
}

static int build_dec_seq_header(uint8_t *pbHeader,
                                const PAYLOAD_TYPE_E p_enType,
                                const AVStream *st, int *sizelength) {
  AVCodecParameters *avc = st->codecpar;

  uint8_t *pbMetaData = avc->extradata;
  int nMetaData = avc->extradata_size;
  uint8_t *p = pbMetaData;
  uint8_t *a = p + 4 - ((long)p & 3);
  uint8_t *t = pbHeader;
  int size;
  int sps, pps, i, nal;

  size = 0;
  *sizelength = 4; // default size length(in bytes) = 4
  if (p_enType == PT_H264) {
    if (nMetaData > 1 && pbMetaData && pbMetaData[0] == 0x01) {
      // check mov/mo4 file format stream
      p += 4;
      *sizelength = (*p++ & 0x3) + 1;
      sps = (*p & 0x1f); // Number of sps
      p++;
      for (i = 0; i < sps; i++) {
        nal = (*p << 8) + *(p + 1) + 2;
        SET_BYTE(t, 0x00);
        SET_BYTE(t, 0x00);
        SET_BYTE(t, 0x00);
        SET_BYTE(t, 0x01);
        SET_BUFFER(t, p + 2, nal - 2);
        p += nal;
        size += (nal - 2 + 4); // 4 => length of start code to be inserted
      }

      pps = *(p++); // number of pps
      for (i = 0; i < pps; i++) {
        nal = (*p << 8) + *(p + 1) + 2;
        SET_BYTE(t, 0x00);
        SET_BYTE(t, 0x00);
        SET_BYTE(t, 0x00);
        SET_BYTE(t, 0x01);
        SET_BUFFER(t, p + 2, nal - 2);
        p += nal;
        size += (nal - 2 + 4); // 4 => length of start code to be inserted
      }
    } else if (nMetaData > 3) {
      size = -1; // return to meaning of invalid stream data;
      for (; p < a; p++) {
        if (p[0] == 0 && p[1] == 0 && p[2] == 1) {
          // find startcode
          size = avc->extradata_size;
          if (!pbHeader || !pbMetaData)
            return 0;
          SET_BUFFER(pbHeader, pbMetaData, size);
          break;
        }
      }
    }
  } else if (p_enType == PT_H265) {
    if (nMetaData > 1 && pbMetaData && pbMetaData[0] == 0x01) {
      static const int8_t nalu_header[4] = {0, 0, 0, 1};
      int numOfArrays = 0;
      uint16_t numNalus = 0;
      uint16_t nalUnitLength = 0;
      uint32_t offset = 0;

      p += 21;
      *sizelength = (*p++ & 0x3) + 1;
      numOfArrays = *p++;

      while (numOfArrays--) {
        p++; // NAL type
        numNalus = (*p << 8) + *(p + 1);
        p += 2;
        for (i = 0; i < numNalus; i++) {
          nalUnitLength = (*p << 8) + *(p + 1);
          p += 2;
          // if(i == 0)
          {
            memcpy(pbHeader + offset, nalu_header, 4);
            offset += 4;
            memcpy(pbHeader + offset, p, nalUnitLength);
            offset += nalUnitLength;
          }
          p += nalUnitLength;
        }
      }

      size = offset;
    } else if (nMetaData > 3) {
      size = -1; // return to meaning of invalid stream data;

      for (; p < a; p++) {
        if (p[0] == 0 && p[1] == 0 && p[2] == 1) // find startcode
        {
          size = avc->extradata_size;
          if (!pbHeader || !pbMetaData)
            return 0;
          SET_BUFFER(pbHeader, pbMetaData, size);
          break;
        }
      }
    }
  } else {
    SET_BUFFER(pbHeader, pbMetaData, nMetaData);
    size = nMetaData;
  }

  return size;
}

int vdec_ChnAttr_init(VDEC_CHN_ATTR_S *pVdecChnAttr, PAYLOAD_TYPE_E enType,
                      int picWidth, int picHeight) {
  // int streambufSize = 0;
  if (pVdecChnAttr == NULL) {
    printf("pVdecChnAttr is NULL!\n");
    return -1;
  }
  // 该步骤必不可少
  memset(pVdecChnAttr, 0, sizeof(VDEC_CHN_ATTR_S));
  // 设置解码模式分别为 PT_H264 PT_H265 PT_MJPEG
  pVdecChnAttr->enType = enType;
  // 设置解码模式为帧模式
  pVdecChnAttr->enMode = VIDEO_MODE_FRAME;
  // 设置像素格式 NV12格式
  pVdecChnAttr->enPixelFormat = HB_PIXEL_FORMAT_NV12;
  // 输入buffer个数
  pVdecChnAttr->u32FrameBufCnt = 3;
  // 输出buffer个数
  pVdecChnAttr->u32StreamBufCnt = 3;
  // 输出buffer size，必须1024对齐
  pVdecChnAttr->u32StreamBufSize =
      (picWidth * picHeight * 3 / 2 + 1024) & ~0x3ff;
  // 使用外部buffer
  pVdecChnAttr->bExternalBitStreamBuf = HB_TRUE;
  if (enType == PT_H265) {
    // 使能带宽优化
    pVdecChnAttr->stAttrH265.bandwidth_Opt = HB_TRUE;
    // 普通解码模式，IPB帧解码
    pVdecChnAttr->stAttrH265.enDecMode = VIDEO_DEC_MODE_NORMAL;
    // 输出顺序按照播放顺序输出
    pVdecChnAttr->stAttrH265.enOutputOrder = VIDEO_OUTPUT_ORDER_DISP;
    // 不启用CLA作为BLA处理
    pVdecChnAttr->stAttrH265.cra_as_bla = HB_FALSE;
    // 配置tempral id为绝对模式
    pVdecChnAttr->stAttrH265.dec_temporal_id_mode = 0;
    // 保持2
    pVdecChnAttr->stAttrH265.target_dec_temporal_id_plus1 = 2;
  }
  if (enType == PT_H264) {
    // 使能带宽优化
    pVdecChnAttr->stAttrH264.bandwidth_Opt = HB_TRUE;
    // 普通解码模式，IPB帧解码
    pVdecChnAttr->stAttrH264.enDecMode = VIDEO_DEC_MODE_NORMAL;
    // 输出顺序按照播放顺序输出
    pVdecChnAttr->stAttrH264.enOutputOrder = VIDEO_OUTPUT_ORDER_DISP;
  }
  if (enType == PT_JPEG) {
    // 使用内部buffer
    pVdecChnAttr->bExternalBitStreamBuf = HB_FALSE;
    // 配置镜像模式，不镜像
    pVdecChnAttr->stAttrJpeg.enMirrorFlip = DIRECTION_NONE;
    // 配置旋转模式，不旋转
    pVdecChnAttr->stAttrJpeg.enRotation = CODEC_ROTATION_0;
    // 配置crop，不启用
    pVdecChnAttr->stAttrJpeg.stCropCfg.bEnable = HB_FALSE;
  }
  return 0;
}

int main(int argc, char **argv) {

  gflags::SetUsageMessage("some usage message");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // 加载模型
  hbPackedDNNHandle_t packed_dnn_handle;
  char const *model_path = FLAGS_model_path.c_str();
  hbDNNInitializeFromFiles(&packed_dnn_handle, &model_path, 1);

  // 获取模型名称
  char const **model_name_list;
  int model_count = 0;
  hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle);

  // 获取dnn_handle
  hbDNNHandle_t dnn_handle;
  hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name_list[0]);

  int s32Ret;
  int picWidth = 1920;
  int picHeight = 1080;
  int inputHeight = 640;
  int inputWidth = 640;
  PAYLOAD_TYPE_E enType = PT_H264;

  // init video pool
  VP_CONFIG_S struVpConf;
  memset(&struVpConf, 0x00, sizeof(VP_CONFIG_S));
  struVpConf.u32MaxPoolCnt = 32;
  HB_VP_SetConfig(&struVpConf);
  s32Ret = HB_VP_Init();
  if (s32Ret != 0) {
    printf("vp_init fail s32Ret = %d !\n", s32Ret);
  }
  // init video decode module
  s32Ret = HB_VDEC_Module_Init();
  if (s32Ret) {
    printf("HB_VDEC_Module_Init: %d\n", s32Ret);
  }

  int vdecChn = 0; // vdecChn channel
  VDEC_CHN_ATTR_S vdecChnAttr;
  s32Ret = vdec_ChnAttr_init(&vdecChnAttr, enType, picWidth, picHeight);
  if (s32Ret) {
    printf("sample_venc_ChnAttr_init failded: %d\n", s32Ret);
  }
  // 创建channel
  s32Ret = HB_VDEC_CreateChn(vdecChn, &vdecChnAttr);
  if (s32Ret != 0) {
    printf("HB_VDEC_CreateChn %d failed, %x.\n", vdecChn, s32Ret);
    return s32Ret;
  }
  // 设置channel属性
  s32Ret = HB_VDEC_SetChnAttr(vdecChn, &vdecChnAttr); // config
  if (s32Ret != 0) {
    printf("HB_VDEC_SetChnAttr failed\n");
    return s32Ret;
  }

  s32Ret = HB_VDEC_StartRecvStream(vdecChn);
  if (s32Ret != 0) {
    printf("HB_VDEC_StartRecvStream failed\n");
    return s32Ret;
  }

  // 准备buffer
  int i = 0;
  char *mmz_vaddr[5];
  for (i = 0; i < 5; i++) {
    mmz_vaddr[i] = NULL;
  }
  uint64_t mmz_paddr[5];
  memset(mmz_paddr, 0, sizeof(mmz_paddr));
  int mmz_size = picHeight * picWidth;
  int mmz_cnt = 5;
  for (i = 0; i < mmz_cnt; i++) {
    s32Ret = HB_SYS_Alloc(&mmz_paddr[i], (void **)&mmz_vaddr[i], mmz_size);
    if (s32Ret == 0) {
      printf("mmzAlloc paddr = 0x%lx, vaddr = 0x%lx i = %d \n",
             (uint64_t)mmz_paddr[i], (uint64_t)mmz_vaddr[i], i);
    }
  }

  // 用于存储码流数据（内部模式直接把码流的buffer地址给到vir_ptr字段就行）
  VIDEO_STREAM_S pstStream;
  // 获取解码之后的数据
  VIDEO_FRAME_S stFrameInfo;
  memset(&pstStream, 0, sizeof(VIDEO_STREAM_S));

  int eos = 0;
  int mmz_index = 0;
  int error = 0;
  int count = 0;
  int videoIndex = -1;
  int seqHeaderSize = 0;
  int firstPacket = 1;
  int bufSize = 0;

  // Step4: 准备输入数据（用于存放yuv数据）
  hbDNNTensor input_tensor;
  // resize后送给bpu运行的图像
  hbDNNTensor input_tensor_resized;

  memset(&input_tensor, '\0', sizeof(hbDNNTensor));
  input_tensor.properties.tensorLayout = HB_DNN_LAYOUT_NCHW;
  // 张量类型为Y通道及UV通道为输入的图片, 方便直接使用 vpu出来的y和uv分离的数据
  // 用于Y和UV分离的场景，主要为我们摄像头数据通路场景
  input_tensor.properties.tensorType = HB_DNN_IMG_TYPE_NV12_SEPARATE;

  // 准备模型输出数据的空间
  int output_count;
  hbDNNGetOutputCount(&output_count, dnn_handle);
  hbDNNTensor *output = new hbDNNTensor[output_count];
  for (int i = 0; i < output_count; i++) {
    hbDNNTensorProperties &output_properties = output[i].properties;
    hbDNNGetOutputTensorProperties(&output_properties, dnn_handle, i);

    // 获取模型输出尺寸
    int out_aligned_size = 4;
    for (int j = 0; j < output_properties.alignedShape.numDimensions; j++) {
      out_aligned_size =
          out_aligned_size * output_properties.alignedShape.dimensionSize[j];
    }
    hbSysMem &mem = output[i].sysMem[0];
    hbSysAllocCachedMem(&mem, out_aligned_size);
  }

  AVFormatContext *avContext = nullptr;
  uint8_t *seqHeader = nullptr;
  AVPacket avpacket = {0};

  AVDictionary *option = 0;
  av_dict_set(&option, "stimeout", "3000000", 0);
  av_dict_set(&option, "bufsize", "1024000", 0);
  av_dict_set(&option, "rtsp_transport", "tcp", 0);

  int try_open_count = 0;
  do {
    s32Ret = avformat_open_input(&avContext, FLAGS_video.c_str(), 0, &option);
    if (s32Ret != 0) {
      printf("avformat_open_input: %d, retry\n", s32Ret);
    }
    printf("try open\n");
  } while (s32Ret != 0 && try_open_count++ < 300);
  s32Ret = avformat_find_stream_info(avContext, 0);
  videoIndex =
      av_find_best_stream(avContext, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
  if (videoIndex == -1) {
    printf("Didn't find a video stream.\n");
    return -1;
  }
  av_init_packet(&avpacket);

  do {
    VDEC_CHN_STATUS_S pstStatus;
    std::cout << "do while" << std::endl;
    HB_VDEC_QueryStatus(vdecChn, &pstStatus);
    if (pstStatus.cur_input_buf_cnt >= (uint32_t)mmz_cnt) {
      usleep(10000);
      continue;
    }
    usleep(20000);
    if (!avpacket.size) {
      error = av_read_frame(avContext, &avpacket);
    }
    if (error < 0) {
      if (error == AVERROR_EOF || avContext->pb->eof_reached == HB_TRUE) {
        printf("There is no more input data, %d!\n", avpacket.size);
      } else {
        printf("Failed to av_read_frame error(0x%08x)\n", error);
      }
      break;
    } else {
      seqHeaderSize = 0;
      mmz_index = count % mmz_cnt;
      if (firstPacket) {
        // 对第一次读取的处理，数据头
        AVCodecParameters *codec;
        int retSize = 0;
        codec = avContext->streams[videoIndex]->codecpar;
        seqHeader = (uint8_t *)malloc(codec->extradata_size + 1024);
        if (seqHeader == NULL) {
          printf("Failed to mallock seqHeader\n");
          eos = 1;
          break;
        }
        memset((void *)seqHeader, 0x00, codec->extradata_size + 1024);
        seqHeaderSize = build_dec_seq_header(
            seqHeader, enType, avContext->streams[videoIndex], &retSize);

        if (seqHeaderSize < 0) {
          printf("Failed to build seqHeader\n");
          eos = 1;
          break;
        }
        firstPacket = 0;
      }
      if (avpacket.size <= mmz_size) {
        if (seqHeaderSize) {
          memcpy((void *)mmz_vaddr[mmz_index], (void *)seqHeader,
                 seqHeaderSize);
          bufSize = seqHeaderSize;
          std::cout << bufSize << std::endl;
        } else {
          memcpy((void *)mmz_vaddr[mmz_index], (void *)avpacket.data,
                 avpacket.size);
          bufSize = avpacket.size;
          std::cout << bufSize << std::endl;
          av_packet_unref(&avpacket);
          avpacket.size = 0;
        }
      } else {
        printf("The external stream buffer is too small!"
               "avpacket.size:%d, mmz_size:%d\n",
               avpacket.size, mmz_size);
        eos = 1;
        break;
      }
      if (seqHeader) {
        free(seqHeader);
        seqHeader = NULL;
      }
    }
    pstStream.pstPack.phy_ptr = mmz_paddr[mmz_index];
    pstStream.pstPack.vir_ptr = mmz_vaddr[mmz_index];
    pstStream.pstPack.pts = count++;
    pstStream.pstPack.src_idx = mmz_index;
    if (!eos) {
      pstStream.pstPack.size = bufSize;
      pstStream.pstPack.stream_end = HB_FALSE;
    } else {
      pstStream.pstPack.size = 0;
      pstStream.pstPack.stream_end = HB_TRUE;
    }
    printf("[pstStream] pts:%ld, vir_ptr:%ld, size:%d\n", pstStream.pstPack.pts,
           (uint64_t)pstStream.pstPack.vir_ptr, pstStream.pstPack.size);
    printf("feed raw data\n");
    s32Ret = HB_VDEC_SendStream(vdecChn, &pstStream, 3000);
    if (s32Ret == -HB_ERR_VDEC_OPERATION_NOT_ALLOWDED ||
        s32Ret == -HB_ERR_VDEC_UNKNOWN) {
      printf("HB_VDEC_SendStream failed\n");
    }

    // 通常和上面的HB_VDEC_SendStream分开线程处理，做成生产消费者模式，这里演示就放一起了
    s32Ret = HB_VDEC_GetFrame(vdecChn, &stFrameInfo, 500);
    if (s32Ret == 0) {
      // NV12 是 YUV420SP 格式
      input_tensor.sysMem[0].virAddr = stFrameInfo.stVFrame.vir_ptr[0];
      input_tensor.sysMem[0].phyAddr = stFrameInfo.stVFrame.phy_ptr[0];
      input_tensor.sysMem[0].memSize =
          stFrameInfo.stVFrame.height * stFrameInfo.stVFrame.width;

      // 填充 input_tensor.data_ext 成员变量， UV 分量
      input_tensor.sysMem[1].virAddr = stFrameInfo.stVFrame.vir_ptr[1];
      input_tensor.sysMem[1].phyAddr = stFrameInfo.stVFrame.phy_ptr[1];
      input_tensor.sysMem[1].memSize =
          stFrameInfo.stVFrame.height * stFrameInfo.stVFrame.width / 2;

      // HB_DNN_IMG_TYPE_NV12_SEPARATE 类型的 layout 为 (1, 3, h, w)
      input_tensor.properties.validShape.numDimensions = 4;
      input_tensor.properties.validShape.dimensionSize[0] = 1;         // N
      input_tensor.properties.validShape.dimensionSize[1] = 3;         // C
      input_tensor.properties.validShape.dimensionSize[2] = picHeight; // H
      input_tensor.properties.validShape.dimensionSize[3] = picWidth;  // W
      input_tensor.properties.alignedShape =
          input_tensor.properties.validShape; // 已满足跨距对齐要求，直接赋值

      // 准备模型输入数据（用于存放模型输入大小的数据）
      input_tensor_resized.properties.tensorLayout = HB_DNN_LAYOUT_NCHW;
      input_tensor_resized.properties.tensorType =
          HB_DNN_IMG_TYPE_NV12_SEPARATE;

      input_tensor_resized.sysMem[0].memSize = inputHeight * inputWidth;
      hbSysMem &itr_mem0 = input_tensor_resized.sysMem[0];
      hbSysAllocMem(&itr_mem0, inputHeight * inputWidth);
      input_tensor_resized.sysMem[1].memSize = inputHeight * inputWidth / 2;
      hbSysMem &itr_mem1 = input_tensor_resized.sysMem[1];
      hbSysAllocMem(&itr_mem1, inputHeight * inputWidth / 2);

      // NCHW
      input_tensor_resized.properties.validShape.numDimensions = 4;
      input_tensor_resized.properties.validShape.dimensionSize[0] = 1;
      input_tensor_resized.properties.validShape.dimensionSize[1] = 3;
      input_tensor_resized.properties.validShape.dimensionSize[2] = inputHeight;
      input_tensor_resized.properties.validShape.dimensionSize[3] = inputWidth;
      // 已满足对齐要求
      input_tensor_resized.properties.alignedShape =
          input_tensor_resized.properties.validShape;

      // 将数据Resize到模型输入大小
      hbDNNResizeCtrlParam ctrl;
      HB_DNN_INITIALIZE_RESIZE_CTRL_PARAM(&ctrl);
      hbDNNTaskHandle_t resize_task_handle;
      HB_CHECK_SUCCESS(hbDNNResize(&resize_task_handle, &input_tensor_resized,
                                   &input_tensor, NULL, &ctrl),
                       "hbDNNResize failed");
      HB_CHECK_SUCCESS(hbDNNWaitTaskDone(resize_task_handle, 0),
                       "hbDNNWaitTaskDone failed");

      HB_CHECK_SUCCESS(hbDNNReleaseTask(resize_task_handle),
                       "hbDNNReleaseTask failed");

      // Step6: 推理模型
      hbDNNTaskHandle_t infer_task_handle = nullptr;
      hbDNNInferCtrlParam infer_ctrl_param;
      HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);
      HB_CHECK_SUCCESS(hbDNNInfer(&infer_task_handle, &output,
                                  &input_tensor_resized, dnn_handle,
                                  &infer_ctrl_param),
                       "infer hbDNNInfer failed");

      // Step7: 等待任务结束
      HB_CHECK_SUCCESS(hbDNNWaitTaskDone(infer_task_handle, 0),
                       "infer hbDNNWaitTaskDone failed");

      // Step8: 解析模型输出
      hbSysFlushMem(&(output->sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
      float *out = reinterpret_cast<float *>(output->sysMem[0].virAddr);
      int *shape = output->properties.validShape.dimensionSize;
      std::cout << shape[0] << ", " << shape[1] << ", " << shape[2]
                << std::endl;

      std::unordered_map<int, std::vector<DetectionRet>> cls2bbox;

      int numAnchors = shape[1];
      int num = shape[2];
      for (int i = 0; i < numAnchors * num; i += num) {
        // std::cout << out[i] << ", " << out[i+1] << ", " << out[i+2] << ", "
        // << out[i+3] << ", " << out[i+4] << ", " << out[i+5] << std::endl;
        if (out[i + 4] <= 0.4) {
          continue;
        }
        DetectionRet det;
        det.confidence = out[i + 4];
        det.classId = std::distance(
            out + i + 5, std::max_element(out + i + 5, out + i + num));

        memcpy(&det, &out[i], 5 * sizeof(float));
        if (cls2bbox.count(det.classId) == 0)
          cls2bbox.emplace(det.classId, std::vector<DetectionRet>());
        cls2bbox[det.classId].push_back(det);
      }

      std::vector<DetectionRet> bboxes;
      nms(bboxes, cls2bbox);

      // rect 还原成原始大小
      renderOriginShape(bboxes, {picWidth, picHeight, 3},
                        {inputWidth, inputHeight, 3});

      for (auto &bbox : bboxes) {
        for (auto c : bbox.bbox) {
          std::cout << c << ", ";
        }
        std::cout << bbox.confidence << std::endl;
      }
      
      HB_VDEC_ReleaseFrame(vdecChn, &stFrameInfo);
    }

  } while (1);
  for (i = 0; i < mmz_cnt; i++) {
    s32Ret = HB_SYS_Free(mmz_paddr[i], mmz_vaddr[i]);
    if (s32Ret == 0) {
      printf("mmzFree paddr = 0x%lx, vaddr = 0x%lx i = %d \n",
             (uint64_t)mmz_paddr[i], (uint64_t)mmz_vaddr[i], i);
    }
  }
  if (avContext)
    avformat_close_input(&avContext);

  // 释放内存
  hbSysFreeMem(&(input_tensor.sysMem[0]));
  hbSysFreeMem(&(input_tensor.sysMem[1]));
  hbSysFreeMem(&(input_tensor_resized.sysMem[0]));
  hbSysFreeMem(&(input_tensor_resized.sysMem[1]));
  hbSysFreeMem(&(output->sysMem[0]));

  // 释放模型
  hbDNNRelease(dnn_handle);
  hbDNNRelease(packed_dnn_handle);

  return 0;
}