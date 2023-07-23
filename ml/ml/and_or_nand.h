#pragma once
#include <cstddef>
float sigmoid_f(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

// OR-gate
//{0, 0, 0},
//{ 1, 0, 1 },
//{ 0, 1, 1 },
//{ 1, 1, 1 }

// AND-Gate NAND gate reverse the final numbers
//{0, 0, 0},
//{ 1, 0, 0 },
//{ 0, 1, 0 },
//{ 1, 1, 1 }

// XOR-GATE will stagnate the learning on a asingle neuron
//{ 0, 0, 0 },
//{ 1, 0, 1 },
//{ 0, 1, 1 },
//{ 1, 1, 0 }

//float train[][3] =
//{
//    {0, 0, 0},
//    {1, 0, 1},
//    {0, 1, 1},
//    {1, 1, 1}
//
//};

typedef float sample[3];

sample or_train[] =
{
    {0, 0, 0},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 1}

};

sample and_train[] =
{
    {0, 0, 0},
    {1, 0, 0},
    {0, 1, 0},
    {1, 1, 1}

};

sample nand_train[] =
{
    {0, 0, 1},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 0}

};

sample xor_train[] =
{
    {0, 0, 0},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 0}

};

//#define train_count (sizeof(train)/sizeof(train[0]))
sample* train = or_train;  // Allows easier switching between sets
size_t train_count = 4;


float cost(float w1, float w2, float b) {
    float result = 0.0f;

    for (size_t i = 0; i < train_count; ++i) {
        float x1 = train[i][0];//x = input
        float x2 = train[i][1];
        float y = sigmoid_f(x1 * w1 + x2 * w2 + b); //w = weight
        float d = y - train[i][2];
        result += d * d; //Square the derivative d*d
        //printf("actual: %f, expected: %f\n", y, train[i][1]);
    }
    result /= train_count;
    return result;
}

float gcost(float w1, float w2, float b, 
        float* dw1, float* dw2, float* db) //Gradient Cost funtion
{
	*dw1 = 0.0f;
	*dw2 = 0.0f;
	*db = 0.0f;

	size_t n = train_count;
	for (size_t i = 0; i < n; ++i)
	{
		float xi = train[i][0];
		float yi = train[i][1];
		float zi = train[i][2];
		
		float ai = sigmoidf(xi * w1 + yi * w2 + b);
		float di = 2.0f * (ai - zi) * ai * (1.0f - ai);
		
		*dw1 += di * xi;
		*dw2 += di * yi;
		*db += di;
	}
    *dw1 /= n;
    *dw2 /= n;
    *db /= n;
}

float rand_float(void)
{
    return (float)rand() / (float)RAND_MAX;
}


//MAIN

srand(time(0));
float w1 = rand_float();
float w2 = rand_float();
float b = rand_float();

float eps = 1e-1f;
float rate = 1e-1f;

for (size_t i = 0; i < 5000; ++i) {
    float c = cost(w1, w2, b);
    //printf("w1 = %f, w2 = %f, b = %f, c = %f\n", w1, w2, b, c);

    float dw1 = (cost(w1 + eps, w2, b) - c) / eps;
    float dw2 = (cost(w1, w2 + eps, b) - c) / eps;
    float db = (cost(w1, w2, b + eps) - c) / eps;
    w1 -= rate * dw1;
    w2 -= rate * dw2;
    b -= rate * db;
}

for (size_t i = 0; i < 2; ++i)
{
    for (size_t j = 0; j < 2; ++j)
    {
        printf("%zu | %zu = %f\n", i, j, sigmoid_f(i * w1 + j * w2 + b));
    }
}


