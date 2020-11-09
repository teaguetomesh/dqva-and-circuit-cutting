#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <stdbool.h>
#include <sys/time.h>
#include <unistd.h>
#include "mkl.h"

#include <immintrin.h>
#include "omp.h"

void build(char* build_file, char* data_folder, char* dest_folder, int rank);
void print_float_arr(float *arr, long long int num_elements);
void print_int_arr(int *arr, int num_elements);
int* get_nonzero_summation_term_idx(char* build_file, char* data_folder, int rank);
float print_log(double log_time, double elapsed_time, int num_finished_jobs, int num_total_jobs, double log_frequency, int rank);
void scopy_sequential(long long int n, float *src, float *dst);
void scopy_par(long long int n, float *src, float *dst);
double get_sec();

int main(int argc, char** argv) {
    char *build_file = argv[1];
    char *data_folder = argv[2];
    char *dest_folder = argv[3];
    int rank = atoi(argv[4]);
    int recursion_layer = atoi(argv[5]);
    
    build(build_file,data_folder,dest_folder,rank);
    printf("recursion_layer %d build rank %d DONE\n",recursion_layer,rank);
    return 0;
}

int* get_nonzero_summation_term_idx(char* build_file, char* data_folder, int rank) {
    int total_active_qubit, num_subcircuits, num_summation_terms, num_cuts;
    FILE* build_fptr = fopen(build_file, "r");
    fscanf(build_fptr,"total_active_qubit=%d num_subcircuits=%d num_summation_terms=%d num_cuts=%d\n",\
    &total_active_qubit,&num_subcircuits,&num_summation_terms,&num_cuts);
    int summation_term_ctr;
    int num_nonzero_summation_terms = 0;
    int *non_zero_summation_term_idx = calloc(num_summation_terms+1,sizeof(int));
    for (summation_term_ctr=0;summation_term_ctr<num_summation_terms;summation_term_ctr++) {
        bool summation_term_is_zero = false;
        int subcircuit_ctr;
        for (subcircuit_ctr=0;subcircuit_ctr<num_subcircuits;subcircuit_ctr++) {
            int subcircuit_idx, subcircuit_kron_index;
            fscanf(build_fptr,"%d,%d ",&subcircuit_idx,&subcircuit_kron_index);
            char *build_data_file = malloc(256*sizeof(char));
            sprintf(build_data_file, "%s/kron_%d_%d.txt", data_folder, subcircuit_idx, subcircuit_kron_index);
            if(access(build_data_file, F_OK) == -1) {
                // file doesn't exist
                summation_term_is_zero = true;
                free(build_data_file);
            }
        }
        if (!summation_term_is_zero) {
            non_zero_summation_term_idx[num_nonzero_summation_terms+1] = summation_term_ctr;
            num_nonzero_summation_terms++;
        }
    }
    fclose(build_fptr);
    non_zero_summation_term_idx[0] = num_nonzero_summation_terms;
    // printf("num_subcircuits %d non_zero_num_summation_terms %d/%d\n",\
    // num_subcircuits,num_nonzero_summation_terms,num_summation_terms);
    return non_zero_summation_term_idx;
}

void build(char* build_file, char* data_folder, char* dest_folder, int rank) {
    int *non_zero_summation_term_idx = get_nonzero_summation_term_idx(build_file,data_folder,rank);
    
    int total_active_qubit, num_subcircuits, num_summation_terms, num_cuts;
    FILE* build_fptr = fopen(build_file, "r");
    fscanf(build_fptr,"total_active_qubit=%d num_subcircuits=%d num_summation_terms=%d num_cuts=%d\n",\
    &total_active_qubit,&num_subcircuits,&num_summation_terms,&num_cuts);
    long long int reconstruction_len = (long long int) pow(2,total_active_qubit);
    float *reconstructed_prob = (float*) calloc(reconstruction_len,sizeof(float));

    // cblas_sger parameters
    MKL_INT incx, incy;
    CBLAS_LAYOUT layout = CblasRowMajor;
    float alpha = 1;
    incx = 1;
    incy = 1;
    
    int summation_term_ctr;
    int non_zero_summation_term_ctr = 1;
    int num_non_zero_summation_terms_remaining = non_zero_summation_term_idx[0];
    double total_build_time = 0;
    double log_time = 0;
    for (summation_term_ctr=0;summation_term_ctr<num_summation_terms;summation_term_ctr++) {
        double build_begin = get_sec();
        if (num_non_zero_summation_terms_remaining==0) {
            // printf("Rank %d : no more remaining non_zero summation terms\n",rank);
            break;
        }
        else if (summation_term_ctr==non_zero_summation_term_idx[non_zero_summation_term_ctr]) {
            // printf("Rank %d : summation term %d is nonzero\n",rank,summation_term_ctr);
            float *summation_term = (float*) calloc(reconstruction_len,sizeof(float));
            
            // Read first subcircuit
            int subcircuit_idx_0, subcircuit_kron_index_0;
            fscanf(build_fptr,"%d,%d ",&subcircuit_idx_0,&subcircuit_kron_index_0);
            // printf("Subcircuit %d, kron term %d\n",subcircuit_idx_0,subcircuit_kron_index_0);
            char *build_data_file_0 = malloc(256*sizeof(char));
            sprintf(build_data_file_0, "%s/kron_%d_%d.txt", data_folder, subcircuit_idx_0, subcircuit_kron_index_0);
            // printf("Reading file %s\n",build_data_file_0);
            FILE* build_data_fptr_0 = fopen(build_data_file_0, "r");
            int num_active_0;
            fscanf(build_data_fptr_0,"num_active %d\n",&num_active_0);
            // printf("num_active %d\n",num_active_0);
            long long int subcircuit_active_len_0 = (long long int) pow(2,num_active_0);
            long long int state_ctr_0;
            for (state_ctr_0=0;state_ctr_0<subcircuit_active_len_0;state_ctr_0++) {
                // printf("Read state %d\n",state_ctr_0);
                fscanf(build_data_fptr_0,"%f ",&summation_term[state_ctr_0]);
            }
            fclose(build_data_fptr_0);
            free(build_data_file_0);

            int subcircuit_ctr;
            long long int summation_term_accumulated_len = subcircuit_active_len_0;
            // print_float_arr(summation_term,summation_term_accumulated_len);
            for (subcircuit_ctr=1;subcircuit_ctr<num_subcircuits;subcircuit_ctr++) {
                int subcircuit_idx, subcircuit_kron_index;
                fscanf(build_fptr,"%d,%d ",&subcircuit_idx,&subcircuit_kron_index);
                char *build_data_file = malloc(256*sizeof(char));
                sprintf(build_data_file, "%s/kron_%d_%d.txt", data_folder, subcircuit_idx, subcircuit_kron_index);
                FILE* build_data_fptr = fopen(build_data_file, "r");
                int num_active;
                fscanf(build_data_fptr,"num_active %d\n",&num_active);
                long long int subcircuit_active_len = (long long int) pow(2,num_active);
                long long int state_ctr;
                float *subcircuit_kron_term = (float*) calloc(subcircuit_active_len,sizeof(float));
                for (state_ctr=0;state_ctr<subcircuit_active_len;state_ctr++) {
                    fscanf(build_data_fptr,"%f ",&subcircuit_kron_term[state_ctr]);
                }

                // print_float_arr(subcircuit_kron_term,subcircuit_active_len);

                float *dummy_summation_term = (float*) calloc(summation_term_accumulated_len*subcircuit_active_len,sizeof(float));
                cblas_sger(layout, summation_term_accumulated_len, subcircuit_active_len, alpha, summation_term, incx, subcircuit_kron_term, incy, dummy_summation_term, subcircuit_active_len);
                summation_term_accumulated_len *= subcircuit_active_len;
                cblas_scopy(summation_term_accumulated_len, dummy_summation_term, 1, summation_term, 1);
                // scopy_par(summation_term_accumulated_len, dummy_summation_term, summation_term);
                free(dummy_summation_term);

                // print_float_arr(summation_term,summation_term_accumulated_len);

                fclose(build_data_fptr);
                free(build_data_file);
                free(subcircuit_kron_term);
            }
            vsAdd(reconstruction_len, reconstructed_prob, summation_term, reconstructed_prob);
            free(summation_term);
            non_zero_summation_term_ctr++;
            num_non_zero_summation_terms_remaining--;
        }
        else {
            // printf("Rank %d : summation term %d is zero\n",rank,summation_term_ctr);
            char line[256];
            fgets(line, sizeof(line), build_fptr);
        }
        log_time += get_sec() - build_begin;
        total_build_time += get_sec() - build_begin;
        log_time = print_log(log_time,total_build_time,summation_term_ctr+1,num_summation_terms,10,rank);
    }

    cblas_sscal(reconstruction_len, pow(0.5,num_cuts), reconstructed_prob, 1);
    // print_float_arr(reconstructed_prob,reconstruction_len);

    char *build_result_file = malloc(256*sizeof(char));
    sprintf(build_result_file, "%s/reconstructed_prob_%d.txt", dest_folder, rank);
    FILE* build_data_fptr = fopen(build_result_file, "w");
    long long int state_ctr;
    for (state_ctr=0;state_ctr<reconstruction_len;state_ctr++) {
        fprintf(build_data_fptr,"%e ",reconstructed_prob[state_ctr]);
    }
    fclose(build_data_fptr);
    free(build_result_file);
    
    fclose(build_fptr);
    free(non_zero_summation_term_idx);
    free(reconstructed_prob);

    // printf("Rank %d build DONE\n", rank);

    char *summary_file = malloc(256*sizeof(char));
    sprintf(summary_file, "%s/rank_%d_summary.txt", dest_folder, rank);
    FILE *summary_fptr = fopen(summary_file, "a");
    fprintf(summary_fptr,"\nTotal build time = %e\n",total_build_time);
    fprintf(summary_fptr,"DONE");
    free(summary_file);
    fclose(summary_fptr);
    return;
}

void scopy_sequential(long long int n, float *src, float *dst) {
    long long int n32 = n & -32;
    long long int i;
    float *src_curr_pos = src, *dst_curr_pos = dst;
    for (i = 0; i < n32; i += 32){
        _mm256_storeu_ps(dst_curr_pos, _mm256_loadu_ps(src_curr_pos));
        _mm256_storeu_ps(dst_curr_pos+8, _mm256_loadu_ps(src_curr_pos+8));
        _mm256_storeu_ps(dst_curr_pos+16, _mm256_loadu_ps(src_curr_pos+16));
        _mm256_storeu_ps(dst_curr_pos+24, _mm256_loadu_ps(src_curr_pos+24));
        src_curr_pos += 32; dst_curr_pos += 32;
    }
    if (n32 == n) return;
    src_curr_pos = src + n32;
    dst_curr_pos = dst + n32;
    for (i = n32; i < n; i++){
        *dst_curr_pos = *src_curr_pos;
        dst_curr_pos++;
        src_curr_pos++;
    }
}

void scopy_par(long long int n, float *src, float *dst) {
    int TOTAL_THREADS=atoi(getenv("OMP_NUM_THREADS"));
    if (TOTAL_THREADS<=1){
        scopy_sequential(n,src,dst);
        return;
    }
    int tid;
    int max_cpu_num=(int)sysconf(_SC_NPROCESSORS_ONLN);
    if (TOTAL_THREADS>max_cpu_num) TOTAL_THREADS=max_cpu_num;
    #pragma omp parallel for schedule(static)
    for (tid = 0; tid < TOTAL_THREADS; tid++){
        long int NUM_DIV_NUM_THREADS = n / TOTAL_THREADS * TOTAL_THREADS;
        long int DIM_LEN = n / TOTAL_THREADS;
        long int EDGE_LEN = (NUM_DIV_NUM_THREADS == n) ? n / TOTAL_THREADS : n - NUM_DIV_NUM_THREADS + DIM_LEN;
        if (tid == 0)
        scopy_sequential(EDGE_LEN,src,dst);
        else
        scopy_sequential(DIM_LEN,src + EDGE_LEN + (tid - 1) * DIM_LEN, dst + EDGE_LEN + (tid - 1) * DIM_LEN);
    }
    return;
}

void print_int_arr(int *arr, int num_elements) {
    int ctr;
    if (num_elements<=10) {
        for (ctr=0;ctr<num_elements;ctr++) {
            printf("%d ",arr[ctr]);
        }
    }
    else {
        for (ctr=0;ctr<5;ctr++) {
            printf("%d ",arr[ctr]);
        }
        printf(" ... ");
        for (ctr=num_elements-5;ctr<num_elements;ctr++) {
            printf("%d ",arr[ctr]);
        }
    }
    printf(" = %d elements\n",num_elements);
}

void print_float_arr(float *arr, long long int num_elements) {
    long long int ctr;
    if (num_elements<=10) {
        for (ctr=0;ctr<num_elements;ctr++) {
            printf("%e ",arr[ctr]);
        }
    }
    else {
        for (ctr=0;ctr<5;ctr++) {
            printf("%e ",arr[ctr]);
        }
        printf(" ... ");
        for (ctr=num_elements-5;ctr<num_elements;ctr++) {
            printf("%e ",arr[ctr]);
        }
    }
    printf(" = %lld elements\n",num_elements);
}

float print_log(double log_time, double elapsed_time, int num_finished_jobs, int num_total_jobs, double log_frequency, int rank) {
    if (log_time>log_frequency) {
        double eta = elapsed_time/num_finished_jobs*num_total_jobs - elapsed_time;
        printf("Rank %d finished building %d/%d, elapsed = %e, ETA = %e\n",rank,num_finished_jobs,num_total_jobs,elapsed_time,eta);
        return 0;
    }
    else {
        return log_time;
    }
}

double get_sec() {
    struct timeval time;
    gettimeofday(&time, NULL);
    return (time.tv_sec + 1e-6 * time.tv_usec);
}
