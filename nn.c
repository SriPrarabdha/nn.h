#include <time.h>
#define NN_IMPLEMENTATION
#include "nn.h"

int main(void){
    srand(time(0));
    Mat m1 = mat_alloc(5,8);
    mat_rand(m1 , 1 , 2);
    mat_print(m1,"m1");
    printf("\n-----------------\n");
    Mat m2 = mat_alloc(5,8);
    // mat_rand(m2 , 1, 10);
    mat_fill(m2 , 1);
    mat_print(m2, "m2");
    printf("\n-----------------\n");
    mat_add(m1,m2);
    mat_print(m1 , "m1");

    float id_data[4] = {1.3 , 2.345 , 5 , 10};
    Mat a = {.rows = 2 , .cols = 2 , .es=id_data};
    mat_print(a, "a");
    mat_transpose(a);
    printf("\n-----------------\n");
    mat_print(a , "a");
    Mat b = mat_alloc(2,2);
    mat_copy(b,a);
    MAT_PRINT(b);
    return 0;
}