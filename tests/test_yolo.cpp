#include "dnn/hb_dnn.h"
#include "dnn/hb_sys.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#define HB_CHECK_SUCCESS(value, errmsg)                                \
	do {                                                               \
		/*value can be call of function*/                                  \
		int ret_code = value;                                           \
		if (ret_code != 0) {                                             \
			printf("[BPU ERROR] %s, error code:%d\n", errmsg, ret_code); \
			return ret_code;                                               \
		}                                                                \
	} while (0);


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

int main(int argc, char **argv) {
  // Step1: 加载模型
  hbPackedDNNHandle_t packed_dnn_handle;
  char const *model_path = argv[1];
  hbDNNInitializeFromFiles(&packed_dnn_handle, &model_path, 1);

  // Step2: 获取模型名称
  char const **model_name_list;
  int model_count = 0;
  hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle);

  // Step3: 获取dnn_handle
  hbDNNHandle_t dnn_handle;
  hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name_list[0]);

  // Step4: 准备输入数据（用于存放yuv数据）
  /*
	hbDNNTensor input_tensor;
	// resize后送给bpu运行的图像
	hbDNNTensor *input_tensor_resized = &handle->m_resized_tensors[handle->m_cur_resized_tensor];

	memset(&input_tensor, '\0', sizeof(hbDNNTensor));
	input_tensor.properties.tensorLayout = HB_DNN_LAYOUT_NCHW;
	// 张量类型为Y通道及UV通道为输入的图片, 方便直接使用 vpu出来的y和uv分离的数据
	input_tensor.properties.tensorType = HB_DNN_IMG_TYPE_NV12_SEPARATE; // 用于Y和UV分离的场景，主要为我们摄像头数据通路场景

	// 填充 input_tensor.sysMem 成员变量 Y 分量
	input_tensor.sysMem[0].virAddr = input_data->addr[0];
	input_tensor.sysMem[0].phyAddr = input_data->paddr[0];
	input_tensor.sysMem[0].memSize = input_data->stride_size * input_data->height;

	// 填充 input_tensor.data_ext 成员变量， UV 分量
	input_tensor.sysMem[1].virAddr = input_data->addr[1];
	input_tensor.sysMem[1].phyAddr = input_data->paddr[1];
	input_tensor.sysMem[1].memSize = (input_data->stride_size * input_data->height) / 2;

#if 0 // test
	if (nv12_index++ % 100 == 0) {
		sprintf(nv12_file_name, "1920x1080_nv12_input_%d.yuv", nv12_index++);
		x3_dumpToFile2plane(nv12_file_name, input_data->addr[0], input_data->addr[1],
			input_tensor.sysMem[0].memSize, input_tensor.sysMem[1].memSize);
	}
#endif

	// HB_DNN_IMG_TYPE_NV12_SEPARATE 类型的 layout 为 (1, 3, h, w)
	input_tensor.properties.validShape.numDimensions = 4;
	input_tensor.properties.validShape.dimensionSize[0] = 1;						// N
	input_tensor.properties.validShape.dimensionSize[1] = 3;						// C
	input_tensor.properties.validShape.dimensionSize[2] = input_data->height;		// H
	input_tensor.properties.validShape.dimensionSize[3] = input_data->stride_size; 	// W
	input_tensor.properties.alignedShape = input_tensor.properties.validShape;		// 已满足跨距对齐要求，直接赋值

	// 准备模型输入数据（用于存放模型输入大小的数据）
	input_tensor_resized->properties.tensorLayout = HB_DNN_LAYOUT_NCHW;
	input_tensor_resized->properties.tensorType = HB_DNN_IMG_TYPE_NV12_SEPARATE;

	// NCHW
	input_tensor_resized->properties.validShape.numDimensions = 4;
	input_tensor_resized->properties.validShape.dimensionSize[0] = 1;
	input_tensor_resized->properties.validShape.dimensionSize[1] = 3;
	input_tensor_resized->properties.validShape.dimensionSize[2] = model_h;
	input_tensor_resized->properties.validShape.dimensionSize[3] = model_w;
	input_tensor_resized->properties.alignedShape = input_tensor_resized->properties.validShape;		// 已满足对齐要求

	// 将数据Resize到模型输入大小
	hbDNNResizeCtrlParam ctrl;
	HB_DNN_INITIALIZE_RESIZE_CTRL_PARAM(&ctrl);
	hbDNNTaskHandle_t task_handle;
	HB_CHECK_SUCCESS(
		hbDNNResize(&task_handle, input_tensor_resized, &input_tensor, NULL, &ctrl),
		"hbDNNResize failed");
  */
  hbDNNTensor input;
  hbDNNTensorProperties input_properties;
  hbDNNGetInputTensorProperties(&input_properties, dnn_handle, 0);
  input.properties = input_properties;
  auto &mem = input.sysMem[0];

  int yuv_length = 640 * 640 * 3;
  hbSysAllocCachedMem(&mem, yuv_length);

  // Step5: 准备模型输出数据的空间
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

  // Step6: 推理模型
  hbDNNTaskHandle_t task_handle = nullptr;
  hbDNNInferCtrlParam infer_ctrl_param;
  HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);
  hbDNNInfer(&task_handle, &output, &input, dnn_handle, &infer_ctrl_param);

  // Step7: 等待任务结束
  hbDNNWaitTaskDone(task_handle, 0);

  // Step8: 解析模型输出
  hbSysFlushMem(&(output->sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);

  float *out = reinterpret_cast<float *>(output->sysMem[0].virAddr);
  int *shape = output->properties.validShape.dimensionSize;
  std::cout << shape[0] << std::endl;
  std::cout << shape[1] << std::endl;
  std::cout << shape[2] << std::endl;

  std::unordered_map<int, std::vector<DetectionRet>> m;
  std::vector<DetectionRet> bboxes;

  int numAnchors = shape[1];
  int num = shape[2];
  for (int i = 0; i < numAnchors * num; i += num) {
    if (out[i + 4] <= 0.3)
      continue;
    DetectionRet det;
    det.classId = std::distance(out + i + 5,
                                std::max_element(out + i + 5, out + i + num));
    int real_idx = i + 5 + det.classId;
    det.confidence = out[real_idx];
    memcpy(&det, &output[i], 5 * sizeof(float));
    if (m.count(det.classId) == 0)
      m.emplace(det.classId, std::vector<DetectionRet>());
    m[det.classId].push_back(det);
  }
  nms(bboxes, m);
  // rect 还原成原始大小
  renderOriginShape(bboxes, {1920, 1080, 3}, {640, 640, 3});

  // 释放内存
  hbSysFreeMem(&(input.sysMem[0]));
  hbSysFreeMem(&(output->sysMem[0]));

  // 释放模型
  hbDNNRelease(packed_dnn_handle);
  hbDNNReleaseTask(task_handle);
  return 0;
}