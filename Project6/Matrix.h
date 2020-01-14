#ifndef MATRIX 
#define MATRIX 
#include <iostream>
class Matrix 	
{

private:
	
public:
	int rows;
	int columns;
	double** values;
	Matrix();
	Matrix(int R,int C);
	Matrix(const Matrix &m2);
	void IdentityMatrix();
	void zeroMatrix();
	double vector_Euclidean_norm();
	static void Jacobi(Matrix &A, Matrix &b, Matrix &x, double eps, int &it_cnt, double &time);
	static void Gauss_Seidel(Matrix &A, Matrix &b, Matrix &x, double eps, int &it_cnt, double &time);
	static void LU_factorization(Matrix &L, Matrix &U);
	static void LU_method(Matrix &L, Matrix &U, Matrix &x, Matrix &b);
	static void FowardSubstitution(Matrix &L, Matrix &y, Matrix &b);
	static void BackwardSubstitution(Matrix &U, Matrix &x, Matrix &y);
	Matrix& operator=(Matrix m2);
	Matrix operator+(const Matrix &m2);
	Matrix operator-(const Matrix &m2);
	Matrix operator*(const Matrix &m2);

	~Matrix();
};

#endif  