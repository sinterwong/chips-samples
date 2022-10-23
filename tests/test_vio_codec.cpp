#include "hb_comm_vdec.h"
#include "hb_comm_venc.h"
#include "hb_comm_video.h"
#include "hb_common.h"
#include "hb_sys.h"
#include "hb_type.h"
#include "hb_vdec.h"
#include "hb_venc.h"
#include "hb_vio_interface.h"
#include "hb_vp_api.h"
#include <cstdint>
#include <fcntl.h>
#include <iostream>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
#include <libavformat/avformat.h>
#include <libavutil/timestamp.h>
#ifdef __cplusplus
}
#endif /* __cplusplus */
// H264 H265 MJPEG

#define SET_BYTE(_p, _b) *_p++ = (unsigned char)_b;

#define SET_BUFFER(_p, _buf, _len)                                             \
  memcpy(_p, _buf, _len);                                                      \
  (_p) += (_len);

struct SAMPLE_VDEC_ATTR_S {
  int picWidth;          /*the attribute of video encoder*/
  int picHeight;         /*the attribute of rate  ctrl*/
  PAYLOAD_TYPE_E enType; /*the attribute of gop*/
  int channel;
  std::string file;
  pthread_mutex_t init_lock;
  pthread_cond_t init_cond;
};

int running = 0;

void signal_handler(int signal_number);

void print_usage();

void *feed_decode_data(void *attr);

void *get_decode_data(void *attr);

int sample_vdec_ChnAttr_init(VDEC_CHN_ATTR_S *pVdecChnAttr,
                             PAYLOAD_TYPE_E enType, int picWidth,
                             int picHeight);

int sample_venc_setRcParam(int chn, VENC_CHN_ATTR_S *pVencChnAttr, int bitRate);

static int build_dec_seq_header(uint8_t *pbHeader,
                                const PAYLOAD_TYPE_E p_enType,
                                const AVStream *st, int *sizelength);

int main(int argc, char **argv) {
  int s32Ret;
  // int opt = 0;
  SAMPLE_VDEC_ATTR_S sample_attr;
  sample_attr.picWidth = 1920;
  sample_attr.picHeight = 1080;
  sample_attr.enType = PT_H264;
  sample_attr.file = argv[1]; // "../datas/vps0_chn1_vps1_chn5_venc0.h264"
  pthread_t producer, consumer;

  signal(SIGINT, signal_handler);

  // 初始化VP
  VP_CONFIG_S struVpConf;
  memset(&struVpConf, 0x00, sizeof(VP_CONFIG_S));
  struVpConf.u32MaxPoolCnt = 32;
  HB_VP_SetConfig(&struVpConf);
  s32Ret = HB_VP_Init();
  if (s32Ret != 0) {
    printf("vp_init fail s32Ret = %d !\n", s32Ret);
  }
  running = 1;
  sample_attr.channel = 0;
  pthread_mutex_init(&sample_attr.init_lock, NULL);
  pthread_cond_init(&sample_attr.init_cond, NULL);
  s32Ret = HB_VDEC_Module_Init();
  if (s32Ret) {
    printf("HB_VDEC_Module_Init: %d\n", s32Ret);
  }

  s32Ret = pthread_create(&consumer, NULL, get_decode_data, &sample_attr);
  if (s32Ret != 0) {
    printf("consumer creat failed\n");
    return 0;
  }
  usleep(1000);
  s32Ret = pthread_create(&producer, NULL, feed_decode_data, &sample_attr);
  if (s32Ret != 0) {
    printf("producer creat failed\n");
    return 0;
  }
  pthread_join(producer, NULL);
  pthread_join(consumer, NULL);
  s32Ret = HB_VP_Exit();
  if (s32Ret == 0) {
    printf("vp exit ok!\n");
  }
  s32Ret = HB_VDEC_Module_Uninit();
  if (s32Ret) {
    printf("HB_VDEC_Module_Uninit: %d\n", s32Ret);
  }

  printf("Done\n");
  return 0;
}

void signal_handler(int signal_number) { running = 0; }

void print_usage() {
  printf("Usage: \n \
./sample_vdec_basic -w width -h height -t ecode_type -f file \
\nencode_type can be: h264,h265,mjpeg\n");
}

void *feed_decode_data(void *attr) {
  int s32Ret;
  SAMPLE_VDEC_ATTR_S *sample_attr = (SAMPLE_VDEC_ATTR_S *)attr;
  VDEC_CHN_ATTR_S vdecChnAttr;
  int picWidth = sample_attr->picWidth;
  int picHeight = sample_attr->picHeight;
  PAYLOAD_TYPE_E enType = sample_attr->enType;
  int vdecChn = sample_attr->channel;

  pthread_mutex_lock(&sample_attr->init_lock);
  // 初始化channel属性
  s32Ret = sample_vdec_ChnAttr_init(&vdecChnAttr, enType, picWidth, picHeight);
  if (s32Ret) {
    printf("sample_venc_ChnAttr_init failded: %d\n", s32Ret);
  }
  // 创建channel
  s32Ret = HB_VDEC_CreateChn(vdecChn, &vdecChnAttr);
  if (s32Ret != 0) {
    printf("HB_VDEC_CreateChn %d failed, %x.\n", vdecChn, s32Ret);
    return NULL;
  }
  // 设置channel属性
  s32Ret = HB_VDEC_SetChnAttr(vdecChn, &vdecChnAttr); // config
  if (s32Ret != 0) {
    printf("HB_VDEC_SetChnAttr failed\n");
    return NULL;
  }
  // VENC_RECV_PIC_PARAM_S pstRecvParam;
  // pstRecvParam.s32RecvPicNum = 0; // unchangable

  s32Ret = HB_VDEC_StartRecvStream(vdecChn);
  if (s32Ret != 0) {
    printf("HB_VDEC_StartRecvStream failed\n");
    return NULL;
  }
  pthread_cond_signal(&sample_attr->init_cond);
  pthread_mutex_unlock(&sample_attr->init_lock);

  int i = 0, error = 0;
  FILE *inFile = NULL;
  FILE *outFile = NULL;
  // VDEC_CHN_ATTR_S hb_VdecChnAttr;
  int width = sample_attr->picWidth;
  int height = sample_attr->picHeight;

  // 准备buffer
  char *mmz_vaddr[5];
  for (i = 0; i < 5; i++) {
    mmz_vaddr[i] = NULL;
  }
  uint64_t mmz_paddr[5];
  memset(mmz_paddr, 0, sizeof(mmz_paddr));
  int mmz_size = width * height;
  int mmz_cnt = 5;
  for (i = 0; i < mmz_cnt; i++) {
    s32Ret = HB_SYS_Alloc(&mmz_paddr[i], (void **)&mmz_vaddr[i], mmz_size);
    if (s32Ret == 0) {
      printf("mmzAlloc paddr = 0x%lx, vaddr = 0x%lx i = %d \n",
             (uint64_t)mmz_paddr[i], (uint64_t)mmz_vaddr[i], i);
    }
  }
  VIDEO_FRAME_S pstFrame;
  VIDEO_STREAM_S pstStream;
  memset(&pstFrame, 0, sizeof(VIDEO_FRAME_S));
  memset(&pstStream, 0, sizeof(VIDEO_STREAM_S));

  AVFormatContext *avContext = NULL;
  AVPacket avpacket = {0};
  int videoIndex;
  uint8_t *seqHeader = NULL;
  int seqHeaderSize = 0;
  // uint8_t *picHeader = NULL;
  // int picHeaderSize;
  int firstPacket = 1;
  int eos = 0;
  int bufSize = 0;
  // int lastFrame = 0;
  int count = 0;
  int mmz_index = 0;
  AVDictionary *option = 0;
  av_dict_set(&option, "stimeout", "3000000", 0);
  av_dict_set(&option, "bufsize", "1024000", 0);
  av_dict_set(&option, "rtsp_transport", "tcp", 0);
  do {
    s32Ret =
        avformat_open_input(&avContext, sample_attr->file.c_str(), 0, &option);
    if (s32Ret != 0) {
      printf("avformat_open_input: %d, retry\n", s32Ret);
    }
    printf("try open\n");
  } while ((s32Ret != 0) && (running == 1));
  s32Ret = avformat_find_stream_info(avContext, 0);
  videoIndex =
      av_find_best_stream(avContext, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
  av_init_packet(&avpacket);

  do {
    VDEC_CHN_STATUS_S pstStatus;
    printf("do while\n");
    HB_VDEC_QueryStatus(vdecChn, &pstStatus);
    if (pstStatus.cur_input_buf_cnt >= (uint32_t)mmz_cnt) {
      usleep(10000);
      continue;
    }
    usleep(20000);
    if (running == 0) {
      break;
    }
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
          memcpy((void *)mmz_vaddr[mmz_index], (void *)seqHeader, seqHeaderSize);
          bufSize = seqHeaderSize;
        } else {
          memcpy((void *)mmz_vaddr[mmz_index], (void *)avpacket.data,
                 avpacket.size);
          bufSize = avpacket.size;
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

  } while (running == 1);
  running = 0;
  for (i = 0; i < mmz_cnt; i++) {
    s32Ret = HB_SYS_Free(mmz_paddr[i], mmz_vaddr[i]);
    if (s32Ret == 0) {
      printf("mmzFree paddr = 0x%lx, vaddr = 0x%lx i = %d \n",
             (uint64_t)mmz_paddr[i], (uint64_t)mmz_vaddr[i], i);
    }
  }
  if (avContext)
    avformat_close_input(&avContext);
  if (inFile)
    fclose(inFile);
  if (outFile)
    fclose(outFile);
  return nullptr;
}

void *get_decode_data(void *attr) {
  int s32Ret;
  SAMPLE_VDEC_ATTR_S *sample_attr = (SAMPLE_VDEC_ATTR_S *)attr;
  // VENC_CHN_ATTR_S vencChnAttr;
  // int picWidth = sample_attr->picWidth;
  // int picHeight = sample_attr->picHeight;
  // PAYLOAD_TYPE_E enType = sample_attr->enType;
  int vdecChn = sample_attr->channel;
  FILE *outFile;
  VIDEO_FRAME_S stFrameInfo;
  int idx = 0;
  struct timeval now;
  struct timespec outtime;

  // wait for set ready
  pthread_mutex_lock(&sample_attr->init_lock);
  gettimeofday(&now, NULL);
  outtime.tv_sec = now.tv_sec + 1;
  outtime.tv_nsec = now.tv_usec * 1000;
  pthread_cond_timedwait(&sample_attr->init_cond, &sample_attr->init_lock,
                         &outtime);
  pthread_mutex_unlock(&sample_attr->init_lock);
  while (running) {
    s32Ret = HB_VDEC_GetFrame(vdecChn, &stFrameInfo, 1000);
    if (s32Ret == 0) {
      if (idx++ % 100 == 0) {
        outFile = fopen("decode.nv12", "wb");
        // NV12是YUV420SP格式
        fwrite(stFrameInfo.stVFrame.vir_ptr[0],
               stFrameInfo.stVFrame.height * stFrameInfo.stVFrame.width, 1,
               outFile);
        fwrite(stFrameInfo.stVFrame.vir_ptr[1],
               stFrameInfo.stVFrame.height * stFrameInfo.stVFrame.width / 2, 1,
               outFile);
        printf("vdec %d, %dx%d\n", idx, stFrameInfo.stVFrame.width,
               stFrameInfo.stVFrame.height);
        fclose(outFile);
      }
      HB_VDEC_ReleaseFrame(vdecChn, &stFrameInfo);
    } else {
      printf("HB_VDEC_GetFrame failed:%d\n", s32Ret);
    }
  }

  return nullptr;
}

int sample_vdec_ChnAttr_init(VDEC_CHN_ATTR_S *pVdecChnAttr,
                             PAYLOAD_TYPE_E enType, int picWidth,
                             int picHeight) {
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
