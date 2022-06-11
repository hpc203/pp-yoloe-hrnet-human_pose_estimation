#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>
#include"utils.h"

using namespace cv;
using namespace std;
using namespace Ort;

typedef struct BoxInfo
{
	int xmin;
	int ymin;
	int xmax;
	int ymax;
	float score;
	string name;
} BoxInfo;

class Human_Pose_Estimation
{
public:
	Human_Pose_Estimation(float confThreshold);
	void detect(Mat& cv_image);
private:
	float confThreshold;

	Mat preprocess(Mat srcimg);
	void normalize_(Mat img);
	const int inpWidth = 640;
	const int inpHeight = 640;
	vector<float> input_image_;
	vector<float> scale_factor = { 1,1 };

	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "pp-yoloe");
	Ort::Session *ort_session = nullptr;
	SessionOptions sessionOptions = SessionOptions();
	vector<char*> input_names;
	vector<char*> output_names;
	vector<vector<int64_t>> output_node_dims; // >=1 outputs

	vector<float> input_person;
	Env env_person = Env(ORT_LOGGING_LEVEL_ERROR, "hrnet");
	Ort::Session *ort_session_person = nullptr;
	SessionOptions sessionOptions_person = SessionOptions();
	vector<char*> input_names_person;
	vector<char*> output_names_person;
	vector<vector<int64_t>> output_node_dims_person;
	const int inpWidth_person = 192;
	const int inpHeight_person = 256;
	Mat resize_image(Mat srcimg, int *newh, int *neww, int *top, int *left);
	const bool keep_ratio = true;
	void normalize_person(Mat img);
	float* key_points(Mat srcimg, Rect box);
	vector<float> center{ 0,0 };
	vector<float> scale{ 0,0 };
};

Human_Pose_Estimation::Human_Pose_Estimation(float confThreshold)
{
	string model_path = "mot_ppyoloe_l_36e_pipeline.onnx";
	std::wstring widestr = std::wstring(model_path.begin(), model_path.end());
	///OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);

	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	ort_session = new Session(env, widestr.c_str(), sessionOptions);
	size_t numInputNodes = ort_session->GetInputCount();
	size_t numOutputNodes = ort_session->GetOutputCount();
	AllocatorWithDefaultOptions allocator;
	for (int i = 0; i < numInputNodes; i++)
	{
		input_names.push_back(ort_session->GetInputName(i, allocator));
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
		output_names.push_back(ort_session->GetOutputName(i, allocator));
		Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		output_node_dims.push_back(output_dims);
	}
	this->confThreshold = confThreshold;

	model_path = "dark_hrnet_w32_256x192.onnx";
	std::wstring widestr_person = std::wstring(model_path.begin(), model_path.end());
	///OrtStatus* status_person = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions_person, 0);

	sessionOptions_person.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	ort_session_person = new Session(env_person, widestr_person.c_str(), sessionOptions_person);
	numInputNodes = ort_session_person->GetInputCount();
	numOutputNodes = ort_session_person->GetOutputCount();
	AllocatorWithDefaultOptions allocator_person;
	for (int i = 0; i < numInputNodes; i++)
	{
		input_names_person.push_back(ort_session_person->GetInputName(i, allocator_person));
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
		output_names_person.push_back(ort_session_person->GetOutputName(i, allocator_person));
		Ort::TypeInfo output_type_info_person = ort_session_person->GetOutputTypeInfo(i);
		auto output_tensor_info_person = output_type_info_person.GetTensorTypeAndShapeInfo();
		auto output_dims_person = output_tensor_info_person.GetShape();
		output_node_dims_person.push_back(output_dims_person);
	}
	output_node_dims_person[0][0] = 1;
	center[0] = (float)this->inpWidth_person*0.5;
	center[1] = (float)this->inpHeight_person*0.5;
	scale[0] = (float)this->inpWidth_person / 200;
	scale[1] = (float)this->inpHeight_person / 200;
}

Mat Human_Pose_Estimation::preprocess(Mat srcimg)
{
	Mat dstimg;
	cvtColor(srcimg, dstimg, COLOR_BGR2RGB);
	resize(dstimg, dstimg, Size(this->inpWidth, this->inpHeight), INTER_LINEAR);
	return dstimg;
}

Mat Human_Pose_Estimation::resize_image(Mat srcimg, int *newh, int *neww, int *top, int *left)
{
	int srch = srcimg.rows, srcw = srcimg.cols;
	*newh = this->inpHeight_person;
	*neww = this->inpWidth_person;
	Mat dstimg;
	if (this->keep_ratio && srch != srcw) 
	{
		float hw_scale = (float)srch / srcw;
		if (hw_scale > 1) {
			*newh = this->inpHeight_person;
			*neww = int(this->inpWidth_person / hw_scale);
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_LINEAR);
			*left = int((this->inpWidth_person - *neww) * 0.5);
			copyMakeBorder(dstimg, dstimg, 0, 0, *left, this->inpWidth_person - *neww - *left, BORDER_CONSTANT, 0);
		}
		else {
			*newh = (int)this->inpHeight_person * hw_scale;
			*neww = this->inpWidth_person;
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_LINEAR);
			*top = (int)(this->inpHeight_person - *newh) * 0.5;
			copyMakeBorder(dstimg, dstimg, *top, this->inpHeight_person - *newh - *top, 0, 0, BORDER_CONSTANT, 0);
		}
	}
	else {
		resize(srcimg, dstimg, Size(*neww, *newh), INTER_LINEAR);
	}
	return dstimg;
}

void Human_Pose_Estimation::normalize_(Mat img)
{
	//img.convertTo(img, CV_32F);
	int row = img.rows;
	int col = img.cols;
	this->input_image_.resize(row * col * img.channels());
	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				float pix = img.ptr<uchar>(i)[j * 3 + c];
				this->input_image_[c * row * col + i * col + j] = pix;
			}
		}
	}
}

void Human_Pose_Estimation::normalize_person(Mat img)
{
	//img.convertTo(img, CV_32F);
	int row = img.rows;
	int col = img.cols;
	this->input_person.resize(row * col * img.channels());
	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				float pix = img.ptr<uchar>(i)[j * 3 + c];
				this->input_person[c * row * col + i * col + j] = pix;
			}
		}
	}
}

float* Human_Pose_Estimation::key_points(Mat srcimg, Rect box)
{
	Mat person_img = srcimg(box);
	int newh = 0, neww = 0, padh = 0, padw = 0;
	Mat dstimg = this->resize_image(person_img, &newh, &neww, &padh, &padw);
	this->normalize_person(dstimg);
	array<int64_t, 4> inputShape{ 1, 3, this->inpHeight_person, this->inpWidth_person };

	auto allocateInfo = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value inputTensor = Value::CreateTensor<float>(allocateInfo, input_person.data(), input_person.size(), inputShape.data(), inputShape.size());

	// 开始推理
	vector<Value> output_tensors_hrnet = ort_session_person->Run(RunOptions{ nullptr }, &input_names_person[0], &inputTensor, 1, output_names_person.data(), output_names_person.size());
	float* floatarr = output_tensors_hrnet[0].GetTensorMutableData<float>();
	float* preds = new float[output_node_dims_person[0][1] * 2 + 2];
	get_final_preds(floatarr, output_node_dims_person[0], center, scale, preds);

	for (int i = 0; i < output_node_dims_person[0][1]; i++)
	{
		preds[i] = box.x + (preds[i] - padw)*box.width / neww;
		preds[i + 17] = box.y + (preds[i + 17] - padh)*box.height / newh;
	}
	return preds;
}

void Human_Pose_Estimation::detect(Mat& srcimg)
{
	Mat dstimg = this->preprocess(srcimg);
	this->normalize_(dstimg);
	array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight, this->inpWidth };
	array<int64_t, 2> scale_shape_{ 1, 2 };

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	vector<Value> ort_inputs;
	ort_inputs.push_back(Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size()));
	ort_inputs.push_back(Value::CreateTensor<float>(allocator_info, scale_factor.data(), scale_factor.size(), scale_shape_.data(), scale_shape_.size()));
	// 开始推理
	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, input_names.data(), ort_inputs.data(), 2, output_names.data(), output_names.size());
	const float* outs = ort_outputs[0].GetTensorMutableData<float>();
	const int* box_num = ort_outputs[1].GetTensorMutableData<int>();

	const float ratioh = float(srcimg.rows) / this->inpHeight;
	const float ratiow = float(srcimg.cols) / this->inpWidth;
	vector<BoxInfo> boxs;
	for (int i = 0; i < box_num[0]; i++)
	{
		if (outs[0] > -1 && outs[1] > this->confThreshold)
		{
			boxs.push_back({ int(outs[2] * ratiow), int(outs[3] * ratioh), int(outs[4] * ratiow), int(outs[5] * ratioh), outs[1], "person" });
		}
		outs += 6;
	}

	for (size_t n = 0; n < boxs.size(); ++n)
	{
		Rect rect;
		rect.x = boxs[n].xmin;
		rect.y = boxs[n].ymin;
		rect.width = boxs[n].xmax - boxs[n].xmin;
		rect.height = boxs[n].ymax - boxs[n].ymin;
		float* points = this->key_points(srcimg, rect);

		rectangle(srcimg, Point(boxs[n].xmin, boxs[n].ymin), Point(boxs[n].xmax, boxs[n].ymax), Scalar(0, 0, 255), 2);
		string label = format("%.2f", boxs[n].score);
		label = boxs[n].name + ":" + label;
		putText(srcimg, label, Point(boxs[n].xmin, boxs[n].ymin - 5), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);

		for (int i = 0; i < 17; i++)
		{
			int x_coord = int(points[i]);
			int y_coord = int(points[i + 17]);
			circle(srcimg, Point2d(x_coord, y_coord), 3, Scalar(0, 255, 0), -1);
		}

		/*for (int i = 0; i < 20; i = i + 2)
		{
			line(srcimg, Point2d(int(points[pair_line[i]]), int(points[pair_line[i] + 17])),
				Point2d(int(points[pair_line[i + 1]]), int(points[pair_line[i + 1] + 17])), (0, 0, 255), 4);
		}*/	
	}
}

int main()
{
	Human_Pose_Estimation mynet(0.7);
	string imgpath = "imgs/person.jpg";
	Mat srcimg = imread(imgpath);
	mynet.detect(srcimg);

	static const string kWinName = "Deep learning object detection in ONNXRuntime";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, srcimg);
	waitKey(0);
	destroyAllWindows();
}