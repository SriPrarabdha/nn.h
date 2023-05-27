#ifndef NN_H_
#define NN_H_

#include <stddef.h>
#include <stdio.h>
#include <math.h>

#ifndef NN_MALLOC
#include <stdlib.h> 
#define NN_MALLOC malloc
#endif // NN_MALLOC

#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert
#endif // NN_ASSERT

typedef struct {
    size_t rows;
    size_t cols;
    size_t stride;
    float *es;
}Mat ;

#define MAT_AT(m, i, j) (m).es[(i)*(m).stride + (j)]

float rand_float(void);
float sigmoidf(float x);

Mat mat_alloc(size_t rows , size_t cols);
void mat_fill(Mat m , float num);
void mat_dot(Mat dest , Mat a , Mat b);
void mat_add(Mat dest , Mat a);
void mat_print(Mat m , const char *name);
void mat_rand(Mat m , float low , float high);
void mat_transpose(Mat m);
void mat_sigmoid(Mat m);
Mat mat_row(Mat m , size_t row);
void mat_copy(Mat dest , Mat src);

#define MAT_PRINT(m) mat_print(m , #m)

#endif // NN_H_

#ifdef NN_IMPLEMENTATION

float rand_float(void){
    return (float) rand() / (float) RAND_MAX;
}

float sigmoidf(float x){
    return 1.f / (1.f + expf(-x));
}

void mat_sigmoid(Mat m){
    for(size_t i = 0; i<m.rows ; ++i){
        for(size_t j = 0; j<m.cols ; ++j){
            MAT_AT(m,i,j) = sigmoidf(MAT_AT(m,i,j));
        }
    }
    (void) m;
}
void mat_transpose(Mat m){
    float swap;
    for(size_t i = 0; i<m.rows ; ++i){
        for(size_t j = i; j<m.cols ; ++j){
            swap = MAT_AT(m , i , j);
            MAT_AT(m , i , j) = MAT_AT(m , j , i);
            MAT_AT(m , j , i) = swap;
        }
    }
    (void) m;
}

void mat_fill(Mat m , float num){
    for(size_t i = 0 ; i<m.rows ; ++i){
        for(size_t j = 0 ; j<m.cols ; ++j){
            MAT_AT(m , i , j) = num;
        }
    }
    (void) m;
}

Mat mat_alloc(size_t rows , size_t cols){
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.es = NN_MALLOC(sizeof(*m.es)*rows*cols);
    NN_ASSERT(m.es != NULL);
    return m;
};
void mat_dot(Mat dest , Mat a , Mat b){
    NN_ASSERT(a.cols == b.rows);
    NN_ASSERT(dest.rows == a.rows);
    NN_ASSERT(dest.cols == b.cols);
    for(size_t i = 0 ; i<dest.rows ; ++i){
        for(size_t j = 0 ; j<dest.cols ; ++j){
            MAT_AT(dest , i , j) = 0;
            for(size_t k = 0 ; k<a.cols ; ++k){
                MAT_AT(dest , i , j) += MAT_AT(a , i, k) * MAT_AT(b, k, j);
            }
        }
    }
    (void) dest;
    (void) a;
    (void) b;
};
void mat_add(Mat dest , Mat a){
    NN_ASSERT(dest.rows == a.rows);
    NN_ASSERT(dest.cols == a.cols);
    for(size_t i = 0 ; i<a.rows ; ++i){
        for(size_t j =0 ; j<a.cols ; ++j){
            MAT_AT(dest, i, j) += MAT_AT(a,i,j);
        }
    }
    (void) dest;
    (void) a;
};
void mat_print(Mat m , const char *name){
    
    printf("%s = [\n" , name);
    for(size_t i=0 ; i<m.rows ; ++i){
        for(size_t j = 0 ; j<m.cols ; ++j){
            printf("    %f " , MAT_AT(m, i, j));
        }
        printf("\n");
    }
    printf("]\n");
    (void) m;
};

void mat_rand(Mat m , float low , float high){
    for(size_t i=0 ; i<m.rows ; ++i){
        for(size_t j = 0 ; j<m.cols ; ++j){
            MAT_AT(m,i,j) = rand_float()*(high - low) + low;
        }
    }
    (void) m;
}

Mat mat_row(Mat m , size_t row){
    return (Mat) {
        .rows = 1 ,
        .cols = m.cols ,
        .stride = m.stride, 
        .es = &MAT_AT(m,row , 0),
    };
}

void mat_copy(Mat dest , Mat src){
    NN_ASSERT(dest.rows == src.rows);
    NN_ASSERT(dest.cols == src.cols);

    for(size_t i = 0 ; i<dest.rows ; ++i){
        for(size_t j = 0 ; j<dest.cols ; ++j){
            MAT_AT(dest , i , j) = MAT_AT(src , i , j);
        }
    }
}


#endif // NN_IMPLEMENTATION