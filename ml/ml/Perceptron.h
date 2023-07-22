#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <time.h>


float train[][2] = {
	{0, 0},
	{1, 2},
	{2, 4},
	{3, 6},
	{4, 8}
};

#define train_count (sizeof(train)/sizeof(train[0]))

float rand_float(void)
{
	return (float)rand() / (float)RAND_MAX;
}

//x1, x2, x3...., bias (Input Neurons)
//w1, w2, w3...     (Weights)
//y = x1*w1 + x2*w2 + x3*w3 + ... + bias

float cost(float w, float b) {
	float result = 0.0f;

	for (size_t i = 0; i < train_count; ++i) {
		float x = train[i][0];
		float y = x * w + b;
		float d = y - train[i][1];
		result += d * d;
		//printf("actual: %f, expected: %f\n", y, train[i][1]);
	}
	result /= train_count;
	return result;
}

//Gradient descent cost
float dcost(float w)
{
	float result = 0.0f;
	size_t n = train_count;
	for (size_t i = 0; i < n; ++i)
	{
		float x = train[i][0];
		float y = train[i][1];
		result += 2 * (x * w - y) * x;
	}
	result /= n;
	return result;
}

//EVERYTHNG FROM INSIDE main()
// 
//srand(time(0));
srand(69);

// y = x * w
float w = rand_float() * 10.0f;
float b = rand_float() * 5.0f;


float eps = 1e-3f;
float rate = 1e-3f;

printf("%f\n", cost(w, b));

for (size_t i = 0; i < 500; ++i) {
	float c = cost(w, b);

	float dw = (cost(w + eps, b) - c) / eps;
	float db = (cost(w, b + eps) - c) / eps;

	w -= rate * dw;
	b -= rate * db;
	printf("index = %i:  Cost = %f, w = %f, b = %f\n", i, c, w, b);
}
printf("-----------------\n");
printf("w = %f, b = %f\n", w, b);