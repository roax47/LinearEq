#include "Matrix.h"
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <ctime>
#include <fstream>

#define N 965

void def_matrix(Matrix &A) {
	//index number = 165256
	int e = 2;
	for (int i = 0; i < A.rows; i++) {
		for (int j = 0; j < A.columns; j++) {
			if (j == i) A.values[i][j] = (double)(5 + e);
			if (j == i - 2 || j == i - 1 || j == i + 1 || j == i + 2) A.values[i][j] = -1.0;
		}
	}
}

void matrix_v2(Matrix &A) {
	for (int i = 0; i < A.rows; i++) {
		for (int j = 0; j < A.columns; j++) {
			if (j == i) A.values[i][j] = 3.0;
			if (j == i - 2 || j == i - 1 || j == i + 1 || j == i + 2) A.values[i][j] = -1.0;
		}
	}
}

void set_b(Matrix &b) {
	int f = 5;
	for (int i = 0; i < b.rows; i++)
		b.values[i][0] = sin(i*(f + 1));
}

int main() {

	//Task A
	Matrix A = Matrix(N, N);
	Matrix b = Matrix(N, 1);
	
	set_b(b);

	//Task B
	int jacobi_iter = 0;
	int gs_iter = 0;
	double jacobi_time = 0.0;
	double gs_time = 0.0;
	def_matrix(A);

	//starting vectors with values 0
	Matrix x_jacobi = Matrix(N, 1);
	Matrix x_gs = Matrix(N, 1);


	Matrix::Jacobi(A, b, x_jacobi, pow(10,-9),jacobi_iter,jacobi_time);
	std::cout << "Jacobi" << std::endl << "-------------" << std::endl << "Iterations: " << jacobi_iter << std::endl << "Execution Time: "
		<< jacobi_time << "ms" << std::endl;
	
	std::ofstream myfile("jacobi.txt");
	if (myfile.is_open())
	{
		for (int i = 0; i < x_jacobi.rows; i++) {
			myfile << x_jacobi.values[i][0] << std::endl;
		}
		myfile.close();
		std::cout << "Values saved to file jacobi.txt" << std::endl;
	}

	Matrix::Gauss_Seidel(A, b, x_gs, pow(10, -9), gs_iter,gs_time);
	std::cout << std::endl << "Gauss Seidel" << std::endl << "-------------" << std::endl << "Iterations: " << gs_iter << std::endl << "Execution Time: "
		<< gs_time << "ms" << std::endl;

	std::ofstream myfile1("gs.txt");
	if (myfile1.is_open())
	{
		for (int i = 0; i < x_gs.rows; i++) {
			myfile1 << x_gs.values[i][0] << std::endl;
		}
		myfile1.close();
		std::cout << "Values saved to file gs.txt" << std::endl << std::endl;
	}

	
	//Task C
	matrix_v2(A);
	jacobi_iter = 0;
	jacobi_time = 0.0;
	Matrix x_j2 = Matrix(N, 1);
	Matrix::Jacobi(A, b, x_j2, pow(10, -9), jacobi_iter,jacobi_time);

	gs_iter = 0;
	gs_time = 0.0;
	Matrix x_gs2 = Matrix(N, 1);
	Matrix::Gauss_Seidel(A, b, x_gs2, pow(10, -9), gs_iter,gs_time);
	

	//Task D

	double LU_time = 0.0;
	matrix_v2(A); //change matrix to one from task C

	Matrix x_LU = Matrix(N, 1);
	Matrix U = A;
	Matrix L = Matrix(U.rows, U.columns);
	L.IdentityMatrix();

	std::clock_t start = std::clock();
	Matrix::LU_factorization(L,U);
	Matrix::LU_method(L, U, x_LU, b);
	LU_time = (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000);

	std::ofstream myfile2("LU.txt");
	if (myfile2.is_open())
	{
		for (int i = 0; i < x_LU.rows; i++) {
			myfile2 << x_LU.values[i][0] << std::endl;
		}
		myfile2.close();
		
	}

	Matrix r = A* x_LU - b;
	double norm = r.vector_Euclidean_norm();
	std::cout << std::endl << "LU method" << std::endl << "-------------" << std::endl << "Excecution time: "
		<< LU_time << "ms" << std::endl << "Norm value: " << norm << std::endl;
	std::cout << "Values saved to file LU.txt" << std::endl << std::endl;



	//Task E
	int N_tab[6] = { 100,500,1000,2000,3000,5000};
	std::cout << "Checking times for N: 100,500,1000,2000,3000,5000" << std::endl;
	std::ofstream file;
	file.open("mn_p2.txt");
	for (int i = 0; i < sizeof(N_tab) / sizeof(int); i++) {

		Matrix A = Matrix(N_tab[i], N_tab[i]);
		def_matrix(A); //set matrix to one from task A
		Matrix x = Matrix(N_tab[i], 1);
		Matrix b = Matrix(N_tab[i], 1);
		set_b(b);
		
		jacobi_iter = 0;
		gs_iter = 0;
		jacobi_time = 0.0;
		gs_time = 0.0;
		LU_time = 0.0;

		Matrix::Jacobi(A, b, x, pow(10, -9), jacobi_iter, jacobi_time);
		x.zeroMatrix();
		Matrix::Gauss_Seidel(A, b, x, pow(10, -9), gs_iter, gs_time);
		x.zeroMatrix();
		Matrix U = A;
		Matrix L = Matrix(U.rows, U.columns);
		L.IdentityMatrix();

		std::clock_t start = std::clock();
		Matrix::LU_factorization(L, U);
		Matrix::LU_method(L, U, x, b);
		LU_time = (std::clock() - start) ;

		if (file.is_open())
		{
			file << N_tab[i] << ":" << std::endl << jacobi_time << std::endl << gs_time << std::endl
				<< LU_time << std::endl;
		}
		else std::cout << "Unable to open file" << std::endl;

		std::cout << N_tab[i] << " done" << std::endl;
	}
	file.close();
	std::cout << "Output in file mn_p2.txt" << std::endl;
	return 0;
}