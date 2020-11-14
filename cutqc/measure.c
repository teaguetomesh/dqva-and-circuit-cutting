#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

int collapse_cluster(FILE *input_fptr, FILE *output_fptr, int rank, int subcircuit_idx, int num_instance, int cluster_circ_size, int **correspondece_map, int num_effective_qubits, int num_collapsed);
float* measure_instance(int subcircuit_circ_size, char** meas, float *unmeasured_prob, int **correspondece_map, int num_effective);
void measure(char *eval_folder, int subcircuit_idx, int num_eval_files, int *eval_files, int rank);
int** effective_full_state_correspondence(int cluster_circ_size, char **meas);
int* decToBinary(int num, int num_digits);
int binaryToDec(int *bin_num, int num_digits);
void print_int_arr(int *arr, int num_elements);
void print_float_arr(float *arr, int num_elements);
int search_element(int *arr, int arr_size, int element);
int combine_effective_O_state(int *bin_effective_state, int num_effective_qubits, int *bin_O_state, int num_O_qubits, int *O_qubit_positions);
float print_log(double log_time, double elapsed_time, int num_finished_jobs, int num_total_jobs, double log_frequency, int rank,int subcircuit_idx);
double get_sec();

int main(int argc, char** argv) {
    int rank = atoi(argv[1]);
    char *eval_folder = argv[2];
    int full_circ_size = atoi(argv[3]);
    int subcircuit_idx = atoi(argv[4]);
    int num_eval_files = atoi(argv[5]);
    int *eval_files = calloc(num_eval_files,sizeof(int));
    int i;
    for (i=0; i<num_eval_files; i++) {
        eval_files[i] = atoi(argv[6+i]);
    }

    measure(eval_folder,subcircuit_idx,num_eval_files,eval_files,rank);
    
    free(eval_files);
    // printf("%s subcircuit %d (%d instances) measure rank %d DONE\n",eval_folder,subcircuit_idx,num_eval_files,rank);
    return 0;
}

void measure(char *eval_folder, int subcircuit_idx, int num_eval_files, int *eval_files, int rank) {
    char *eval_file = malloc(256*sizeof(char));
    sprintf(eval_file, "%s/raw_%d_%d.txt", eval_folder, subcircuit_idx, eval_files[0]);
    FILE* eval_fptr = fopen(eval_file, "r");
    int subcircuit_circ_size, num_effective;
    fscanf(eval_fptr, "d=%d effective=%d\n", &subcircuit_circ_size,&num_effective);
    char *init[subcircuit_circ_size], *meas[subcircuit_circ_size];
    int qubit_ctr;
    for (qubit_ctr=0;qubit_ctr<subcircuit_circ_size;qubit_ctr++) {
        init[qubit_ctr] = malloc(16*sizeof(char));
        fscanf(eval_fptr, "%s ", init[qubit_ctr]);
    }
    for (qubit_ctr=0;qubit_ctr<subcircuit_circ_size;qubit_ctr++) {
        meas[qubit_ctr] = malloc(16*sizeof(char));
        fscanf(eval_fptr, "%s ", meas[qubit_ctr]);
    }
    free(eval_file);
    fclose(eval_fptr);
    int **correspondece_map = effective_full_state_correspondence(subcircuit_circ_size, meas);

    int eval_file_ctr;
    double total_measure_time = 0;
    double log_time = 0;
    for (eval_file_ctr=0;eval_file_ctr<num_eval_files;eval_file_ctr++) {
        double measure_begin = get_sec();
        char *eval_file = malloc(256*sizeof(char));
        sprintf(eval_file, "%s/raw_%d_%d.txt", eval_folder, subcircuit_idx, eval_files[eval_file_ctr]);
        // printf("Measuring %s\n",eval_file);
        FILE* eval_fptr = fopen(eval_file, "r");
        char line[256];
        int subcircuit_circ_size, num_effective;
        fscanf(eval_fptr, "d=%d effective=%d\n", &subcircuit_circ_size,&num_effective);
        char *init[subcircuit_circ_size], *meas[subcircuit_circ_size];
        int qubit_ctr;
        for (qubit_ctr=0;qubit_ctr<subcircuit_circ_size;qubit_ctr++) {
            init[qubit_ctr] = malloc(16*sizeof(char));
            fscanf(eval_fptr, "%s ", init[qubit_ctr]);
            // printf("%s ",init[qubit_ctr]);
        }
        for (qubit_ctr=0;qubit_ctr<subcircuit_circ_size;qubit_ctr++) {
            meas[qubit_ctr] = malloc(16*sizeof(char));
            fscanf(eval_fptr, "%s ", meas[qubit_ctr]);
            // printf("%s ",meas[qubit_ctr]);
        }
        long long int state_ctr;
        long long int unmeasured_len = (long long int) pow(2,subcircuit_circ_size);
        float *unmeasured_prob = malloc(unmeasured_len*sizeof(float));
        for (state_ctr=0;state_ctr<unmeasured_len;state_ctr++){
            fscanf(eval_fptr, "%f ", &unmeasured_prob[state_ctr]);
        }
        // printf("\n");
        float* measured_prob = measure_instance(subcircuit_circ_size,meas,unmeasured_prob,correspondece_map,num_effective);
        remove(eval_file);
        free(eval_file);
        fclose(eval_fptr);

        long long int num_effective_states = (long long int) pow(2,num_effective);
        char *meas_file = malloc(256*sizeof(char));
        sprintf(meas_file, "%s/measured_%d_%d.txt", eval_folder, subcircuit_idx, eval_files[eval_file_ctr]);
        FILE *meas_fptr = fopen(meas_file, "w");
        // fprintf(meas_fptr,"effective=%d\n",num_effective);
        for (state_ctr=0;state_ctr<num_effective_states;state_ctr++) {
            fprintf(meas_fptr,"%e ",measured_prob[state_ctr]);
        }
        free(meas_file);
        fclose(meas_fptr);
        log_time += get_sec() - measure_begin;
        total_measure_time += get_sec() - measure_begin;
        // NOTE: log_frequency is hard coded here
        log_time = print_log(log_time,total_measure_time,eval_file_ctr+1,num_eval_files,300,rank,subcircuit_idx);
    }
    char *summary_file = malloc(256*sizeof(char));
    sprintf(summary_file, "%s/rank_%d_summary.txt", eval_folder, rank);
    FILE *summary_fptr = fopen(summary_file, "w");
    fprintf(summary_fptr,"Total measure time = %e\n",total_measure_time);
    fprintf(summary_fptr,"measure DONE\n");
    free(summary_file);
    fclose(summary_fptr);
    return;
}

float* measure_instance(int subcircuit_circ_size, char** meas, float *unmeasured_prob, int **correspondece_map, int num_effective) {
    int num_O_qubits = subcircuit_circ_size - num_effective;
    // printf("\n");
    if (num_effective==subcircuit_circ_size) {
        return unmeasured_prob;
    }
    else{
        long long int measured_len = (long long int) pow(2,num_effective);
        float *measured_prob = calloc(measured_len,sizeof(float));
        long long int measured_state_ctr;
        //#pragma omp parallel for
        for (measured_state_ctr=0;measured_state_ctr<measured_len;measured_state_ctr++) {
            // printf("Effective_state : %d\n",effective_state_ctr);
            int O_state_ctr;
            int num_O_states = (int) pow(2,num_O_qubits);
            for (O_state_ctr=0;O_state_ctr<num_O_states;O_state_ctr++){
                int full_state = correspondece_map[measured_state_ctr][O_state_ctr];
                int *bin_full_state = decToBinary(full_state, subcircuit_circ_size); // Decompose the function to in-place
                int sigma = 1;
                int qubit_ctr;
                for (qubit_ctr=0;qubit_ctr<subcircuit_circ_size;qubit_ctr++) {
                    if (bin_full_state[qubit_ctr]==1 && strcmp(meas[subcircuit_circ_size-1-qubit_ctr],"I")!=0 && strcmp(meas[subcircuit_circ_size-1-qubit_ctr],"comp")!=0) {
                        sigma *= -1;
                    }
                }
                // print_int_arr(bin_full_state, subcircuit_circ_size);
                // printf("(%d) ",full_state);
                measured_prob[measured_state_ctr] += sigma*unmeasured_prob[full_state];
                // printf("corresponding full_state : %d, sigma = %d, val = %.5e, measured_prob = %.5e\n",full_state, sigma, sigma*unmeasured_prob[full_state],measured_prob[measured_state_ctr]);
            }
            if (measured_prob[measured_state_ctr]>10) {
                printf("Something Wrong\n");
                exit(0);
            }
            // printf("\n");
        }
        return measured_prob;
    }
}

int** effective_full_state_correspondence(int cluster_circ_size, char **meas) {
    int num_effective_qubits = 0;
    int num_O_qubits = 0;
    int qubit_ctr;
    int O_qubit_positions[cluster_circ_size];
    for (qubit_ctr=0;qubit_ctr<cluster_circ_size;qubit_ctr++) {
        if (strcmp(meas[qubit_ctr],"comp")==0) {
            num_effective_qubits++;
        }
        else {
            O_qubit_positions[num_O_qubits] = qubit_ctr;
            num_O_qubits++;
        }
    }
    int num_O_states = (int) pow(2,num_O_qubits);
    int num_effective_states = (int) pow(2,num_effective_qubits);
    int effective_state;
    int **correspondece_map = (int **)malloc(sizeof(int *)*num_effective_states);
    for (effective_state=0;effective_state<num_effective_states;effective_state++) {
        int *bin_effective_state = decToBinary(effective_state, num_effective_qubits);
        // printf("Effective state = %d\n",effective_state);
        int O_state;
        correspondece_map[effective_state]=(int *)malloc(sizeof(int)*num_O_states);
        for (O_state=0;O_state<num_O_states;O_state++) {
            int *bin_O_state = decToBinary(O_state, num_O_qubits);
            int full_state = combine_effective_O_state(bin_effective_state, num_effective_qubits, bin_O_state, num_O_qubits, O_qubit_positions);
            // printf("%d ",full_state);
            correspondece_map[effective_state][O_state] = full_state;
        }
        // printf("\n");
    }
    return correspondece_map;
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

int binaryToDec(int *bin_num, int num_digits) {
    int i;
    int dec = 0;
    for (i=0;i<num_digits;i++) {
        if (bin_num[i]==1) {
            // printf("Add %d\n",1<<(num_digits-1-i));
            dec += 1<<(num_digits-1-i);
        }
    }
    return dec;
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

void print_float_arr(float *arr, int num_elements) {
    int ctr;
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
    printf(" = %d elements\n",num_elements);
}

int search_element(int *arr, int arr_size, int element) {
    int i;
    for(i=0;i<arr_size;i++) {
        if (arr[i]==element) {
            return i;
        }
    }
    return -1;
}

int combine_effective_O_state(int *bin_effective_state, int num_effective_qubits, int *bin_O_state, int num_O_qubits, int *O_qubit_positions) {
    // printf("effective_state : ");
    // print_int_arr(bin_effective_state,num_effective_qubits);
    // printf(", inserting O_state ");
    // print_int_arr(bin_O_state,num_O_qubits);
    // printf(" at O positions ");
    // print_int_arr(O_qubit_positions,num_O_qubits);
    // printf("\n");
    int bin_full_state[num_effective_qubits+num_O_qubits];
    int full_state_ctr;
    int effective_state_ctr = 0;
    int O_state_ctr = 0;
    for (full_state_ctr=0;full_state_ctr<num_effective_qubits+num_O_qubits;full_state_ctr++) {
        int O_qubit_position = search_element(O_qubit_positions, num_O_qubits, full_state_ctr);
        if (O_qubit_position==-1) {
            bin_full_state[num_effective_qubits+num_O_qubits-1-full_state_ctr] = bin_effective_state[num_effective_qubits - 1 - effective_state_ctr];
            effective_state_ctr++;
        }
        else {
            bin_full_state[num_effective_qubits+num_O_qubits-1-full_state_ctr] = bin_O_state[O_qubit_position];
        }
    }
    int full_state = binaryToDec(bin_full_state,num_effective_qubits+num_O_qubits);
    // printf("Full state:");
    // print_int_arr(bin_full_state,num_effective_qubits+num_O_qubits);
    // printf(" --> %d\n",full_state);
    return full_state;
}

float print_log(double log_time, double elapsed_time, int num_finished_jobs, int num_total_jobs, double log_frequency, int rank,int subcircuit_idx) {
    if (log_time>log_frequency) {
        double eta = elapsed_time/num_finished_jobs*num_total_jobs - elapsed_time;
        printf("Meas_rank %d measured subcircuit %d %d/%d, elapsed = %e, ETA = %e\n",rank,subcircuit_idx,num_finished_jobs,num_total_jobs,elapsed_time,eta);
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
