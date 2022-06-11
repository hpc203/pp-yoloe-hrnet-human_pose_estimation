#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

vector<float> get_3rd_point(vector<float>& a, vector<float>& b);

vector<float> get_dir(float src_point_x, float src_point_y, float rot_rad);

void affine_tranform(float pt_x, float pt_y, Mat& t, float* x, int p, int num);

Mat get_affine_transform(vector<float>& center, vector<float>& scale, float rot, vector<int>& output_size, int inv);

void transform_preds(float* coords, vector<float>& center, vector<float>& scale, vector<int>& output_size, vector<int64_t>& t, float* target_coords);

void box_to_center_scale(Rect box, int width, int height, vector<float> &center, vector<float> &scale);

void get_max_preds(float* heatmap, vector<int64_t>& t, float* preds, float* maxvals);

void get_final_preds(float* heatmap, vector<int64_t>& t, vector<float>& center, vector<float> scale, float* preds);

int sign(float x);

const int pair_line[] = {
	//0,2,
	//2,4,
	//4,6,
	6,8,
	8,10,
	6,12,
	12,14,
	14,16,

	//0,1,
	//1,3,
	//3,5,
	5,7,
	7,9,
	5,11,
	11,13,
	13,15,
};