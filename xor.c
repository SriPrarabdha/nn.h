#include <time.h>

#define NN_IMPLEMENTATION
#include "nn.h"

typedef struct {
    Mat a0;
    Mat w1,b1, a1;
    Mat w2, b2 , a2;
}Xor;

void forward(Xor m){

    mat_dot(m.a1 , m.a0 , m.w1);
    mat_add(m.a1 , m.b1);
    mat_sigmoid(m.a1);

    mat_dot(m.a2 , m.a1 , m.w2);
    mat_add(m.a2 , m.b2);
    mat_sigmoid(m.a2);

}

float cost(Xor m , Mat ti , Mat to){
    assert(ti.rows = to.rows);
    assert(to.cols == m.a2.cols);
    size_t n = ti.rows;

    float c = 0;
    for (size_t i = 0 ; i<n ; ++i){
        Mat x = mat_row(ti , i);
        Mat y = mat_row(to , i);

        mat_copy(m.a0 , x);
        forward(m);

        size_t q = to.cols;
        for(size_t j = 0; j<q ; ++j){
            float d = MAT_AT(m.a2 , 0 , j) - MAT_AT(y , 0 , j);
            c+=d*d;
        }

    }
    return c/n;
}

float td[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1 ,
    1, 1, 0 ,
};

int main(void){
    srand(time(0));

    size_t stride = 3;
    size_t n = sizeof(td)/sizeof(td[0])/stride;
    Mat ti = {
        .rows = n ,
        .cols = 2 ,
        .stride = stride ,
        .es = td,
    };
    Mat to = {
        .rows = n , 
        .cols = 1,
        .stride = stride ,
        .es = td+2,
    };
    Xor m;

    MAT_PRINT(ti);
    MAT_PRINT(to);

    m.a0 = mat_alloc(1 , 2);

    m.w1 = mat_alloc(2,2);
    m.b1= mat_alloc(1,2);
    m.a1 = mat_alloc(1,2);

    m.w2 = mat_alloc(2,1);
    m.b2 = mat_alloc(1,1);
    m.a2 = mat_alloc(1,1);

    mat_rand(m.w1, 0 , 1);
    mat_rand(m.b1 , 0 , 1);
    mat_rand(m.w2 , 0 , 1);
    mat_rand(m.b2 ,0 , 1);

    forward(m);

    // MAT_PRINT(m.w1);
    // MAT_PRINT(m.b1);
    // MAT_PRINT(m.w2);
    // MAT_PRINT(m.b2);

    for (size_t i = 0 ; i<2 ; ++i){
        for(size_t j = 0 ; j<2 ; ++j){
            MAT_AT(m.a0 , 0 , 0) = i;
            MAT_AT(m.a0 , 0 , 1) = j;

            forward(m);
            float y = *m.a2.es;
            printf("%zu ^ %zu = %f\n" , i , j , y);
        }
    }

    printf("The cost of the model is %f\n" , cost(m,ti,to));
    


    return 0;
}