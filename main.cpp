#include <fstream>
#include <vector>
#include <chrono>
#include <sstream>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include "mpi.h"
using namespace std;

#define TAG_DATA 0

#define MATRIX_SIZE 10000
#define KERNEL_SIZE 3
#define CLAMP(a, b, c) std::min(std::max(a, b), c)
#define CELL_POS(x, Mx, y, My) (CLAMP(x, 0, Mx-1) + Mx * CLAMP(y, 0, My-1))

void generateMatrix(int width, int height, float* &data);
//void generateMatrixFile(int width, int height);
//void readMatrixFile(int& w, int& h, float* &data);
float applyFilter(float* matrix, int mWidth, int mHeight,
                  float* filter, int fWidth, int fHeight,
                  int cellX, int cellY);
float* makeConvolution( float* matrix, int mWidth, int mHeight,
                        float* filter, int fWidth, int fHeight);

void master(int rank, int nproc)
{
	/*
	int numFilas=100;
	int numColumnas=100;

	int* mat1=(int*)malloc(sizeof(int)*numFilas*numColumnas);
	int* mat2=(int*)malloc(sizeof(int)*numFilas*numColumnas);
	int* matRes=(int*)malloc(sizeof(int)*numFilas*numColumnas);

	int operacion=OP_MUL;
	MPI_Status status;

	int subMatrizFilas=numFilas/(nproc-1);
        int resto=numFilas%(nproc-1);
	subMatrizFilas++;
	for(int i=0;i<numFilas;i++)
		for(int j=0;j<numColumnas;j++)
		{
			mat1[i*numFilas + j]=1;
			mat2[i*numFilas + j]=1;
		}

	*/

    int w  = MATRIX_SIZE, h  = MATRIX_SIZE, 
		kw = KERNEL_SIZE, kh = KERNEL_SIZE;
    float* mat = nullptr;
    auto start = std::chrono::system_clock::now();

    generateMatrix(w = MATRIX_SIZE, h = MATRIX_SIZE, mat);
    float* matRes = new float[w*h];

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end - start;
    printf("Matrix generated in: %f s\n", diff.count());

    float filter[KERNEL_SIZE][KERNEL_SIZE] = {
        {1.0f, 2.0f, 1.0f},
        {2.0f, 4.0f, 2.0f},
        {1.0f, 2.0f, 1.0f}
    };


#if MATRIX_SIZE <= 15
    for (int i = 0; i < w; ++i) {
        for (int j = 0; j < h; ++j) {
            printf("%f ", mat[i * w + j]);
        }
        printf("\n");
    }
#endif

	int subH = h/(nproc-1);
    int rest = h%(nproc-1);

	int halfKw = kw/2, halfKh = kh/2;
	int indexCount=0;
	for(int slave = 1; slave < nproc; slave++)
	{
		if ((slave-1) == rest) subH--;

		int sendH = (subH+2*halfKh), sendW = w;

		MPI_Send(&sendW, 1, MPI_INT, slave, TAG_DATA, MPI_COMM_WORLD);
		MPI_Send(&sendH, 1, MPI_INT, slave, TAG_DATA, MPI_COMM_WORLD);
		MPI_Send(&kw,    1, MPI_INT, slave, TAG_DATA, MPI_COMM_WORLD);
		MPI_Send(&kh,    1, MPI_INT, slave, TAG_DATA, MPI_COMM_WORLD);


		MPI_Send(&(mat[indexCount]), sendW*sendH, MPI_FLOAT, slave, TAG_DATA, MPI_COMM_WORLD);
		MPI_Send(filter, kw*kh, MPI_FLOAT, slave, TAG_DATA, MPI_COMM_WORLD);
		
		indexCount+=sendW*sendH;
	}

	subMatrizFilas++;
	int indexCount=0;
	for(int slave=1;slave<nproc;slave++)
	{
		if((slave-1)==resto) subH--;
		int sendH = (subH+2*halfKh), sendW = w;

		MPI_Recv(&(matRes[indexCount]),sendH*sendW, MPI_FLOAT, slave, TAG_DATA, PI_COMM_WORLD, &status);

		indexCount+=sendW*sendH;
	}

	printf("MASTER: matriz convoluted:\n");
}
void slave(int rank, int nproc)
{
	int w = 0;
	int h = 0;
	int kw = 0;
	int kh = 0;
	float* mat;
	float* filter;
	float* matRes;

	MPI_Status status;
	MPI_Recv(&w,  1, MPI_FLOAT, 0, TAG_DATA, MPI_COMM_WORLD, &status);
	MPI_Recv(&h,  1, MPI_FLOAT, 0, TAG_DATA, MPI_COMM_WORLD, &status);
	MPI_Recv(&kw, 1, MPI_FLOAT, 0, TAG_DATA, MPI_COMM_WORLD, &status);
	MPI_Recv(&kh, 1, MPI_FLOAT, 0, TAG_DATA, MPI_COMM_WORLD, &status);

	mat = (float*)malloc(w*h*sizeof(float));
	filter = (float*)malloc(kw*kh*sizeof(float));

	MPI_Recv(mat, w*h, MPI_FLOAT, 0, TAG_DATA, MPI_COMM_WORLD, &status);
	MPI_Recv(filter, kw*kh, MPI_FLOAT, 0, TAG_DATA, MPI_COMM_WORLD, &status);

	matRes = (float*)malloc(w*h*sizeof(float));
	
    printf("SLAVE %d: started\n", nproc);
    start = std::chrono::system_clock::now();

    float* result = makeConvolution(mat, w, h, filter, kw, kh);

    end = std::chrono::system_clock::now();
    diff = end - start;
    printf("Matrix convoluted in: ");
    printf("%f s\n", diff.count());


	MPI_Send(matRes,numFilasM1*numColumnasM2,MPI_INT,0,
			TAG_DATO,MPI_COMM_WORLD);

	char* nombreFich=(char*)malloc(1000);
}


int main_MPI(int argc,char** argv)
{
	int rank;
	int nproc;
	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&nproc);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	switch(rank)
	{
		case 0: master(argc,argv,rank, nproc);
			break;
		default:
			esclavo(argc,argv,rank, nproc);
			break;
	};
	MPI_Finalize();
	return 0;

}

int main_NoMPI() {
    //generateMatrixFile(MATRIX_SIZE, MATRIX_SIZE);

    int w, h;
    float* mat = nullptr;
    //readMatrixFile(w, h, mat);
    auto start = std::chrono::system_clock::now();

    generateMatrix(w = MATRIX_SIZE, h = MATRIX_SIZE, mat);

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end - start;
    printf("Matrix generated in: %f s\n", diff.count());

    float filter[3][3] = {
        {1.0f, 2.0f, 1.0f},
        {2.0f, 4.0f, 2.0f},
        {1.0f, 2.0f, 1.0f }
    };



#if MATRIX_SIZE <= 15
    for (int i = 0; i < w; ++i) {
        for (int j = 0; j < h; ++j) {
            printf("%f ", mat[i * w + j]);
        }
        printf("\n");
    }
#endif

    printf("\n");
    printf("\n");
    
    printf("started\n");
    start = std::chrono::system_clock::now();

    float* result = makeConvolution(mat, w, h, &filter[0][0], 3, 3);

    end = std::chrono::system_clock::now();
    diff = end - start;
    printf("Matrix convoluted in: ");
    printf("%f s\n", diff.count());

#if MATRIX_SIZE <= 15
    for (int i = 0; i < w; ++i) {
        for (int j = 0; j < h; ++j) {
            printf("%f ", result[i * w + j]);
        }
    	printf("\n");
    }
#endif

    return 0;
}

int main(int argc,char** argv) {
	return main_MPI(int argc,char** argv);
	//return main_NoMPI(int argc,char** argv);
}
/*
void readMatrixFile(int& w, int& h, float* &data) {
    ifstream ifs;
    stringstream ss;
    ss << "../matrix/matrix" << MATRIX_SIZE << ".bin";

    ifs.open(ss.str().c_str(), ofstream::in | ofstream::binary);
    if (!ifs.is_open()) {
        cerr << "ERROR: CANNOT OPEN READ FILE";
    }
    float n;

    ifs >> w;
    ifs >> h;
    cout << w << endl;
    cout << h << endl;

    data = new float[w*h];

    cout << "Reading matrix" << endl;
    int k = 0;
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            ifs >> n;
            data[k++] = n;
        }
        cout << ((float)(i*100)/h) << "%\r";
    }
    ifs.close();
    cout << "100%             " << endl;
    cout << "Matrix read" << endl;
}


void generateMatrixFile(int width, int height) {
    ofstream ofs;
    stringstream ss;
    ss << "../matrix/matrix" << width << ".bin";
    ofs.open(ss.str().c_str(), ofstream::out | ofstream::binary | ofstream::trunc);

    ofs << width << endl;
    ofs << height << endl;

    cout << "Generating matrix" << endl;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            ofs << (rand() % 1000) << endl;
            //if (i % MATRIX_SIZE == 0) cout << (i / MATRIX_SIZE) << endl;
        }
        cout << ((float)(i * 100) / height) << "%\r";
    }
    ofs.close();
    cout << "100%               " << endl;
    cout << "Matrix generated" << endl;
}
*/

void generateMatrix(int w, int h, float* &data) {

    data = new float[w*h];

    printf("Generating matrix\n");
    long int k = 0, l = 0;
    #pragma omp parallel for
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            data[k++] = (float)(rand() % 1000);
        }
        stringstream ss;
        ss << ((float)(l++ * 100) / h) << "%\r";
        printf("%s", ss.str().c_str());
    }
    printf("100.00%%\n");
    printf("Matrix generated\n");
}



float applyFilter(float* matrix, int mWidth, int mHeight,
                  float* filter, int fWidth, int fHeight,
                  int cellX, int cellY) {

    float sum = 0.0f;
    int fwr = fWidth / 2;
    int fhr = fHeight / 2;
    int x = cellX - fwr;
    int y = cellY - fhr;
    float currValue;

    for (int i = 0; i < fHeight; ++i) {
        for (int j = 0; j < fWidth; ++j) {
            int pos = CELL_POS((x + i), mHeight, (y + j), mWidth);
            currValue = matrix[pos];
            sum += currValue * filter[i + j * fWidth];
        }
    }
    return sum;
}

float* makeConvolution( float* matrix, int mWidth, int mHeight,
                        float* filter, int fWidth, int fHeight) {

    float filterSum = 0.0f;
    float* data = new float[mWidth*mHeight];
    for (int i = 0; i < fHeight; ++i) {
        for (int j = 0; j < fWidth; ++j) {
            filterSum += filter[i + j * fHeight];
        }
    }

    int k = 0;
    #pragma omp parallel for
    for (int i = 0; i < mHeight; ++i) {
        for (int j = 0; j < mWidth; ++j) {
            data[CELL_POS(i, mHeight, j, mWidth)] = applyFilter(matrix, mWidth, mHeight, filter, fWidth, fHeight, i, j) / filterSum;
        }
        stringstream ss;
        ss << ((float)(k++ * 100) / mHeight) << "%\r";
        printf("%s", ss.str().c_str());
    }

    printf("100.00%%\n");

    return data;
}
