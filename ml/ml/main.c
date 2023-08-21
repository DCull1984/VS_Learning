#define NN_IMPLEMENTATION

#include "nn.h"

float td_xor[] = {
	0,0,0,
	0,1,1,
	1,0,1,
	1,1,0,
};
float td_or[] = {
	0,0,0,
	0,1,1,
	1,0,1,
	1,1,1,
};

float td_and[] = {
	0,0,0,
	0,1,0,
	1,0,0,
	1,1,1,
};

float td_nand[] = {
	0,0,1,
	0,1,1,
	1,0,1,
	1,1,0,
};

int main(void)
{
	srand(_time32(0)); //time shows size warning, using forced 32 bit _time32 

	float* td = td_xor;

	size_t stride = 3;
	size_t n = 4;  //sizeof(td) / sizeof(td[0]) / stride; //Gives the amount of samples

	Mat ti = {
		.rows = n,
		.cols = 2,
		.stride = stride,
		.es = td
	};

	Mat to = {
		.rows = n,
		.cols = 1,
		.stride = stride,
		.es = td + 2
	};

	size_t arch[] = { 2,2,1 }; //input, hidden, output, Xor gates need at least 1 hidden layer
	NN nn = nn_alloc(arch, ARRAY_LEN(arch));
	NN g = nn_alloc(arch, ARRAY_LEN(arch));
	nn_rand(nn, 0, 1);

	
	float rate = 1.f;
	
	printf("cost = %f\n", nn_cost(nn, ti, to));

	//Training the network
	for (size_t i = 0; i < 25000; ++i)
	{
#if 0
		//Eps = epsilon the amount the weights will adjust 1e-1f -> 1e-3f. Adjusting this and rate will affect training speed
		float eps = 1e-1f;
		nn_finite_diff(nn, g, eps, ti, to);
#else
		nn_backprop(nn, g, ti, to);
#endif
		//NN_PRINT(g);
		nn_learn(nn, g, rate);

	}
	printf("cost = %f\n", nn_cost(nn, ti, to));
	
	//NN_PRINT(nn); //Prints the neural nodes

	printf("<--------------->\n");

	for (size_t i = 0; i < 2; ++i)
	{
		for (size_t j = 0; j < 2; ++j)
		{
			MAT_AT(NN_INPUT(nn), 0, 0) = i;
			MAT_AT(NN_INPUT(nn), 0, 1) = j;
			nn_forward(nn);
			printf("%zu ^ %zu = %f\n", i, j, MAT_AT(NN_OUTPUT(nn), 0, 0));
		}
	}

	return 0;

}