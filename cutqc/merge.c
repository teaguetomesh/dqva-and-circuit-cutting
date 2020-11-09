#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <stdbool.h>
#include <sys/time.h>
#include <unistd.h>
#include "mkl.h"

void print_float_arr(float *arr, long long int num_elements);
void merge(char *merge_file, char *data_folder, char *dest_folder, int rank);
int* decToBinary(int num, int num_digits);
double get_sec();
float print_log(double log_time, double elapsed_time, int num_finished_jobs, int num_total_jobs, double log_frequency, int rank);

int main(int argc, char** argv) {
    char *merge_file = argv[1];
    char *data_folder = argv[2];
    char *dest_folder = argv[3];
    int rank = atoi(argv[4]);
    int recursion_layer = atoi(argv[5]);
    merge(merge_file, data_folder, dest_folder, rank);
    printf("recursion_layer %d merge rank %d DONE\n",recursion_layer,rank);
    return 0;
}

void merge(char *merge_file, char *data_folder, char *dest_folder, int rank) {
    int num_files_to_merge;
    FILE* merge_fptr = fopen(merge_file, "r");
    fscanf(merge_fptr,"num_files_to_merge=%d\n",&num_files_to_merge);
    // printf("Rank %d has %d files to merge\n",rank,num_files_to_merge);
    int merge_ctr = 0;
    double total_merge_time = 0;
    double log_time = 0;
    for (merge_ctr=0;merge_ctr<num_files_to_merge;merge_ctr++) {
        double merge_begin = get_sec();
        // subcircuit_idx=0 subcircuit_kron_index=44
        int subcircuit_idx, subcircuit_kron_index, num_effective, num_active;
        fscanf(merge_fptr,"subcircuit_idx=%d subcircuit_kron_index=%d num_effective=%d num_active=%d\n",\
        &subcircuit_idx,&subcircuit_kron_index,&num_effective,&num_active);
        int* qubit_states = (int *) calloc(num_effective,sizeof(int));
        int* active_qubit_basis = (int *) calloc(num_effective,sizeof(int));
        int active = 0;
        int merged = 0;
        int qubit_ctr;
        for (qubit_ctr=0;qubit_ctr<num_effective;qubit_ctr++) {
            fscanf(merge_fptr,"%d ",&qubit_states[qubit_ctr]);
            if (qubit_states[qubit_ctr]==-2) {
                merged++;
            }
            else if (qubit_states[qubit_ctr]==-1) {
                active_qubit_basis[qubit_ctr] = (int) pow(2,(num_active-1-active));
                active++;
            }
        }
        long long int active_len = (long long int) pow(2,active);
        long long int merged_len = (long long int) pow(2,merged);
        long long int effective_len = (long long int) pow(2,num_effective);

        char *data_file = malloc(256*sizeof(char));
        sprintf(data_file, "%s/kron_%d_%d.txt", data_folder, subcircuit_idx, subcircuit_kron_index);
        FILE *data_fptr = fopen(data_file, "r");
        int data_num_effective;
        fscanf(data_fptr,"num_effective %d\n",&data_num_effective);
        assert((num_effective==data_num_effective));

        float* merged_subcircuit_output = (float *) calloc(active_len,sizeof(float));
        long long int effective_state_ctr;
        for (effective_state_ctr=0;effective_state_ctr<effective_len;effective_state_ctr++) {
            int *bin_effective_state = decToBinary(effective_state_ctr, num_effective);
            int merged_state = 0;
            int merge_qubit_ctr;
            for (merge_qubit_ctr=0;merge_qubit_ctr<num_effective;merge_qubit_ctr++) {
                if (qubit_states[merge_qubit_ctr]==-2) continue; // merged
                else if (qubit_states[merge_qubit_ctr]==-1) {
                    // TODO: check if MSB is correct
                    merged_state += bin_effective_state[merge_qubit_ctr]*active_qubit_basis[merge_qubit_ctr]; // active
                }
                else if (qubit_states[merge_qubit_ctr]!=bin_effective_state[merge_qubit_ctr]) { // zoomed
                    merged_state = -1;
                    break;
                }
                else continue;
            }
            // printf("Full state %d --> effective state %d\n",state_ctr,merged_state);
            float unmerged_p;
            fscanf(data_fptr,"%f ",&unmerged_p);
            if (merged_state!=-1) {
                merged_subcircuit_output[merged_state] += unmerged_p;
            }
        }
        fclose(data_fptr);
        free(data_file);

        char *merged_file = malloc(256*sizeof(char));
        sprintf(merged_file, "%s/kron_%d_%d.txt", dest_folder, subcircuit_idx, subcircuit_kron_index);
        FILE *merged_fptr = fopen(merged_file, "w");
        fprintf(merged_fptr,"num_active %d\n",active);
        long long int active_state_ctr;
        for (active_state_ctr=0;active_state_ctr<active_len;active_state_ctr++) {
            fprintf(merged_fptr,"%e ",merged_subcircuit_output[active_state_ctr]);
        }
        fclose(merged_fptr);
        free(merged_file);
        if (active<num_effective) {
            total_merge_time += get_sec() - merge_begin;
        }
        log_time += get_sec() - merge_begin;
        log_time = print_log(log_time,total_merge_time,merge_ctr+1,num_files_to_merge,300,rank);
    }
    fclose(merge_fptr);
    
    char *summary_file = malloc(256*sizeof(char));
    sprintf(summary_file, "%s/rank_%d_summary.txt", dest_folder, rank);
    FILE *summary_fptr = fopen(summary_file, "w");
    fprintf(summary_fptr,"Total merge time = %e\n",total_merge_time);
    fprintf(summary_fptr,"DONE");
    free(summary_file);
    fclose(summary_fptr);
    return;
}

int* decToBinary(int num, int num_digits) {
    int *bin = malloc(num_digits*sizeof(int));
    int i;
    for (i = num_digits - 1; i >= 0; i--) { 
        int k = num >> i;
        if (k & 1) {
            bin[num_digits - 1 - i] = 1;
        }
        else {
            bin[num_digits - 1 - i] = 0;
        }
    }
    return bin;
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
        printf("Rank %d finished merging %d/%d, elapsed = %e, ETA = %e\n",rank,num_finished_jobs,num_total_jobs,elapsed_time,eta);
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