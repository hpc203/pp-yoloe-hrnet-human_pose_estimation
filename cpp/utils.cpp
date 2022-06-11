#include"utils.h"

vector<float> get_3rd_point(vector<float>& a, vector<float>& b) {
	vector<float> direct{ a[0] - b[0],a[1] - b[1] };
	return vector<float>{b[0] - direct[1], b[1] + direct[0]};
}

vector<float> get_dir(float src_point_x, float src_point_y, float rot_rad) {
	float sn = sin(rot_rad);
	float cs = cos(rot_rad);
	vector<float> src_result{ 0.0,0.0 };
	src_result[0] = src_point_x * cs - src_point_y * sn;
	src_result[1] = src_point_x * sn + src_point_y * cs;
	return src_result;
}

void affine_tranform(float pt_x, float pt_y, Mat& t, float* x, int p, int num) {
	float new1[3] = { pt_x, pt_y, 1.0 };
	Mat new_pt(3, 1, t.type(), new1);
	Mat w = t * new_pt;
	x[p] = w.at<float>(0, 0);
	x[p + num] = w.at<float>(1, 0);
}

Mat get_affine_transform(vector<float>& center, vector<float>& scale, float rot, vector<int>& output_size, int inv) {
	vector<float> scale_tmp;
	scale_tmp.push_back(scale[0] * 200);
	scale_tmp.push_back(scale[1] * 200);
	float src_w = scale_tmp[0];
	int dst_w = output_size[0];
	int dst_h = output_size[1];
	float rot_rad = rot * 3.1415926535 / 180;
	vector<float> src_dir = get_dir(0, -0.5 * src_w, rot_rad);
	vector<float> dst_dir{ 0.0, float(-0.5) * dst_w };
	vector<float> src1{ center[0] + src_dir[0],center[1] + src_dir[1] };
	vector<float> dst0{ float(dst_w * 0.5),float(dst_h * 0.5) };
	vector<float> dst1{ float(dst_w * 0.5) + dst_dir[0],float(dst_h * 0.5) + dst_dir[1] };
	vector<float> src2 = get_3rd_point(center, src1);
	vector<float> dst2 = get_3rd_point(dst0, dst1);
	if (inv == 0) {
		float a[6][6] = { {center[0],center[1],1,0,0,0},
						  {0,0,0,center[0],center[1],1},
						  {src1[0],src1[1],1,0,0,0},
						  {0,0,0,src1[0],src1[1],1},
						  {src2[0],src2[1],1,0,0,0},
						  {0,0,0,src2[0],src2[1],1} };
		float b[6] = { dst0[0],dst0[1],dst1[0],dst1[1],dst2[0],dst2[1] };
		Mat a_1 = Mat(6, 6, CV_32F, a);
		Mat b_1 = Mat(6, 1, CV_32F, b);
		Mat result;
		solve(a_1, b_1, result, 0);
		Mat dst = result.reshape(0, 2);
		return dst;
	}
	else {
		float a[6][6] = { {dst0[0],dst0[1],1,0,0,0},
						  {0,0,0,dst0[0],dst0[1],1},
						  {dst1[0],dst1[1],1,0,0,0},
						  {0,0,0,dst1[0],dst1[1],1},
						  {dst2[0],dst2[1],1,0,0,0},
						  {0,0,0,dst2[0],dst2[1],1} };
		float b[6] = { center[0],center[1],src1[0],src1[1],src2[0],src2[1] };
		Mat a_1 = Mat(6, 6, CV_32F, a);
		Mat b_1 = Mat(6, 1, CV_32F, b);
		Mat result;
		solve(a_1, b_1, result, 0);
		Mat dst = result.reshape(0, 2);
		return dst;
	}
}


void transform_preds(float* coords, vector<float>& center, vector<float>& scale, vector<int>& output_size, vector<int64_t>& t, float* target_coords) {
	Mat tran = get_affine_transform(center, scale, 0, output_size, 1);
	for (int p = 0; p < t[1]; ++p) 
	{
		affine_tranform(coords[p], coords[p + t[1]], tran, target_coords, p, t[1]);
	}
}

void box_to_center_scale(Rect box, int width, int height, vector<float> &center, vector<float> &scale) {
	int box_width = box.width;
	int box_height = box.height;
	center[0] = box.x + box_width * 0.5;
	center[1] = box.y + box_height * 0.5;
	float aspect_ratio = width * 1.0 / height;
	int pixel_std = 200;
	if (box_width > aspect_ratio * box_height) {
		box_height = box_width * 1.0 / aspect_ratio;
	}
	else if (box_width < aspect_ratio * box_height) {
		box_width = box_height * aspect_ratio;
	}
	scale[0] = box_width * 1.0 / pixel_std;
	scale[1] = box_height * 1.0 / pixel_std;
	if (center[0] != -1) {
		scale[0] = scale[0] * 1.25;
		scale[1] = scale[1] * 1.25;
	}
}

/*
* 该函数暂时只实现了batch为1的情况
*/
void get_max_preds(float* heatmap, vector<int64_t>& t, float* preds, float* maxvals) {
	int batch_size = t[0];
	int num_joints = t[1];
	int width = t[3];
	float* pred_mask = new float[num_joints * 2];
	int* idx = new int[num_joints * 2];
	for (int i = 0; i < batch_size; ++i) {
		for (int j = 0; j < num_joints; ++j) {
			float max = heatmap[i * num_joints * t[2] * t[3] + j * t[2] * t[3]];
			int max_id = 0;
			for (int k = 1; k < t[2] * t[3]; ++k) {
				int index = i * num_joints * t[2] * t[3] + j * t[2] * t[3] + k;
				if (heatmap[index] > max) {
					max = heatmap[index];
					max_id = k;
				}
			}
			maxvals[j] = max;
			idx[j] = max_id;
			idx[j + num_joints] = max_id;
		}
	}
	for (int i = 0; i < num_joints; ++i) {
		idx[i] = idx[i] % width;
		idx[i + num_joints] = idx[i + num_joints] / width;
		if (maxvals[i] > 0) {
			pred_mask[i] = 1.0;
			pred_mask[i + num_joints] = 1.0;
		}
		else {
			pred_mask[i] = 0.0;
			pred_mask[i + num_joints] = 0.0;
		}
		preds[i] = idx[i] * pred_mask[i];
		preds[i + num_joints] = idx[i + num_joints] * pred_mask[i + num_joints];
	}

}

void get_final_preds(float* heatmap, vector<int64_t>& t, vector<float>& center, vector<float> scale, float* preds) {
	float* coords = new float[t[1] * 2];
	float* maxvals = new float[t[1]];
	int heatmap_height = t[2];
	int heatmap_width = t[3];
	get_max_preds(heatmap, t, coords, maxvals);
	for (int i = 0; i < t[0]; ++i) {
		for (int j = 0; j < t[1]; ++j) {
			int px = int(coords[i * t[1] + j] + 0.5);
			int py = int(coords[i * t[1] + j + t[1]] + 0.5);
			int index = (i * t[1] + j) * t[2] * t[3];
			if (px > 1 && px < heatmap_width - 1 && py>1 && py < heatmap_height - 1) {
				float diff_x = heatmap[index + py * t[3] + px + 1] - heatmap[index + py * t[3] + px - 1];
				float diff_y = heatmap[index + (py + 1) * t[3] + px] - heatmap[index + (py - 1) * t[3] + px];
				coords[i * t[1] + j] += sign(diff_x) * 0.25;
				coords[i * t[1] + j + t[1]] += sign(diff_y) * 0.25;
			}
		}
	}
	vector<int> img_size{ heatmap_width,heatmap_height };
	transform_preds(coords, center, scale, img_size, t, preds);
}

int sign(float x) {
	int w = 0;
	if (x > 0) {
		w = 1;
	}
	else if (x == 0) {
		w = 0;
	}
	else {
		w = -1;
	}
	return w;
}