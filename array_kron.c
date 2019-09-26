#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

#define num_elements 10000

float arr_a[num_elements], arr_b[num_elements], arr_c[num_elements*num_elements];

int main(int argc, char **argv)
{
    int i,j,k;
    int repeats = 10;
    float x,entry;
    float a = 100.0;
    double time_spent = 0.0;
    for (i=0;i<num_elements;i++){
        arr_a[i] = (float)rand()/(float)(RAND_MAX/a);
        arr_b[i] = (float)rand()/(float)(RAND_MAX/a);
    }
    
    clock_t begin = clock();
    for (k=0;k<repeats;k++){
        for (i=0; i<num_elements; i++) {
            for (j=0; j<num_elements; j++) {
                entry = arr_a[i]*arr_b[i];
                arr_c[i*num_elements+j] = entry;
            }
        }
    }
    clock_t end = clock();
    time_spent += (double)(end-begin)/CLOCKS_PER_SEC;
    printf("C time elapsed = %f seconds\n",time_spent);
    return 0;
}