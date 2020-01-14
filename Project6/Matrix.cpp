#include "Matrix.h"
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <ctime>

Matrix::Matrix(int R, int C)
{
	this->rows = R;
	this->columns = C;
	this->values = new double *[rows];
	for (int i = 0; i < rows; i++) {
		values[i] = new double[columns];
		for (int j = 0; j < columns; j++) 
			values[i][j] = 0.0;
	}
		
}	
Matrix::Matrix()
{
	this->rows = 0;
	this->columns = 0;
	this->values = NULL;
	
}
Matrix::Matrix(const Matrix &m2)
{ 
	this->rows = m2.rows;
	this->columns = m2.columns;
	this->values = new double *[rows];
	for (int i = 0; i < rows; i++) {
		values[i] = new double[columns];
		for (int j = 0; j < columns; j++)
			values[i][j] = m2.values[i][j];
	}
}

void Matrix::IdentityMatrix()
{
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < columns; j++)
			if (j == i) values[i][j] = 1.0;
			else values[i][j] = 0.0;
	}
}

void Matrix::zeroMatrix()
{
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < columns; j++)
			values[i][j] = 0.0;
	}
}


double Matrix::vector_Euclidean_norm() {
	double norm = 0.0;
	for (int i = 0; i < rows; i++) {
		norm += pow(values[i][0],2);
	}
	return sqrt(norm);
}


void Matrix::Jacobi(Matrix &A, Matrix &b, Matrix &x,double eps,int &it_cnt, double &time) {

	Matrix r = Matrix(x.rows, 1);
	Matrix x_prev = Matrix(x.rows, 1);
	std::clock_t start = std::clock();
	double new_value;
	int iter = 0;
	while (true) {
		iter++;
		x_prev = x;
		for (int i = 0; i < x.rows; i++) {
			new_value = 0.0;
			for (int j = 0; j < i ; j++) {
				new_value += A.values[i][j] * x_prev.values[j][0];
			}
			for (int k = i + 1; k < x.rows; k++) {
				new_value += A.values[i][k] * x_prev.values[k][0];
			}
			x.values[i][0] = (b.values[i][0] - new_value) / A.values[i][i];
		}
		r = A * x - b;
		if (r.vector_Euclidean_norm() < eps) break;
		if (iter>=2000) {
			std::cout << "Function does not converge" << std::endl;
			break;
		}
		it_cnt++;
	}
	time = (std::clock() - start);
}


void Matrix::Gauss_Seidel(Matrix &A, Matrix &b, Matrix &x, double eps, int &it_cnt, double &time) {

	Matrix r = Matrix(x.rows, 1);
	double new_value;
	std::clock_t start = std::clock();
	int iter = 0;
	while (true) {
		iter++;
		for (int i = 0; i < x.rows; i++) {
			new_value = 0.0;
			for (int j = 0; j < i; j++) {
				new_value += A.values[i][j] * x.values[j][0];
			}
			for (int k = i + 1; k < x.rows; k++) {
				new_value += A.values[i][k] * x.values[k][0];
			}
			x.values[i][0] = (b.values[i][0] - new_value) / A.values[i][i];
		}
		r = A * x - b;
		if (r.vector_Euclidean_norm() < eps) break;
		if (iter >= 2000) {
			std::cout << "Function does not converge" << std::endl;
			break;
		}
		it_cnt++;
	}
	time = (std::clock() - start);
}

void Matrix::LU_factorization(Matrix &L, Matrix &U)
{
	for (int i = 0; i < U.rows; i++) {
		for (int j = i + 1; j < U.columns; j++) {
			L.values[j][i] = U.values[j][i] / U.values[i][i];
			for (int k = i; k < U.columns; k++) {
				U.values[j][k] -= L.values[j][i] * U.values[i][k];
			}
		}
	}
}

void Matrix::LU_method(Matrix &L, Matrix &U, Matrix &x, Matrix &b)
{
	Matrix y = Matrix(x.rows, x.columns);
	FowardSubstitution(L,y,b);
	BackwardSubstitution(U,x,y);
}

void Matrix::FowardSubstitution(Matrix &L, Matrix &y, Matrix &b)
{
	double sum = 0.0;
	for (int i = 0; i < y.rows; i++) {
		sum = 0.0;
		for (int j = 0; j < i; j++) {
			sum += L.values[i][j] * y.values[j][0];
		}
		y.values[i][0] = (b.values[i][0] - sum) / L.values[i][i];
	}
}

void Matrix::BackwardSubstitution(Matrix & U, Matrix & x, Matrix & y)
{
	double sum = 0.0;
	for (int i = x.rows - 1; i >= 0; i--) {
		sum = 0.0;
		for (int j = i + 1; j <=  x.rows - 1; j++) {
			sum += U.values[i][j] * x.values[j][0];
		}
		x.values[i][0] = (y.values[i][0] - sum) / U.values[i][i];
	}
}

Matrix& Matrix::operator=(Matrix m2)
{
	if (this->columns == m2.columns && this->rows == m2.rows) {
		for (int i = 0; i < rows; i++)
			for (int j = 0; j < columns; j++)
				values[i][j] = m2.values[i][j];
		return *this;
	}
	else {
		for (int i = 0; i < rows; ++i)
			delete[] values[i];
		delete[] values;
		this->rows = m2.rows;
		this->columns = m2.columns;
		this->values = new double *[rows];
		for (int i = 0; i < rows; i++) {
			values[i] = new double[columns];
			for (int j = 0; j < columns; j++)
				values[i][j] = m2.values[i][j];
		}
	}
	return *this;
}



Matrix Matrix::operator+(const Matrix & m2)
{
	assert(this->columns == m2.columns);
	assert(this->rows == m2.rows);
	Matrix m = Matrix(m2.rows, m2.columns);

	for (int i = 0; i < m.rows; i++)
		for (int j = 0; j < m.columns; j++)
			m.values[i][j] = this->values[i][j] + m2.values[i][j];

	return m;
}

Matrix Matrix::operator-(const Matrix & m2)
{
	assert(this->columns == m2.columns);
	assert(this->rows == m2.rows);
	Matrix m = Matrix(m2.rows,m2.columns);

	for (int i = 0; i < m.rows; i++) 
		for (int j = 0; j < m.columns; j++)
			m.values[i][j] = this->values[i][j] - m2.values[i][j];

	return m;
}

Matrix Matrix::operator*(const Matrix & m2)
{
	assert(this->columns == m2.rows);
	Matrix m = Matrix(this->rows, m2.columns);
	for (int i = 0; i < m.rows; i++) {
		for (int j = 0; j < m.columns; j++) {
			for (int k = 0; k < this->columns; k++){
				m.values[i][j] += this->values[i][k] * m2.values[k][j];
			}
		}
	}
	return m;
}

Matrix::~Matrix() {
	for (int i=0; i < rows; ++i) 
		delete[] values[i];
	delete[] values;
	values = NULL;
}
