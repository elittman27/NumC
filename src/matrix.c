#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * __m256d _mm256_add_pd (__m256d a, __m256d b)
 * __m256d _mm256_sub_pd (__m256d a, __m256d b)
 * __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
 * __m256d _mm256_mul_pd (__m256d a, __m256d b)
 * __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
 * __m256d _mm256_and_pd (__m256d a, __m256d b)
 * __m256d _mm256_max_pd (__m256d a, __m256d b)
*/

/* Generates a random double between low and high */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/* Generates a random matrix */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
}

void printmatrix(matrix *input){
    printf("\n-----------\n");
    if (input == NULL) {
        printf("Input is NULL\n");
        return;
    }
    printf("Num rows: %d\n", input->rows);
    printf("Num cols: %d\n", input->cols);
    printf("Ref cnt: %d\n", input->ref_cnt);
    printf("Parent: %p\n", input->parent);
    int i, j;
    for (i =0; i < input->rows; i ++){
        for (j=0; j<input->cols; j++){
            printf("%lf ", (input->data)[i*input->cols+j]);
        }
        printf("\n");
    }
    printf("-----------\n\n");
}

/*
 * Returns the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid. Note that the matrix is in row-major order.
 */
double get(matrix *mat, int row, int col) {
    return (mat -> data)[row*(mat -> cols) + col];
}

/*
 * Sets the value at the given row and column to val. You may assume `row` and
 * `col` are valid. Note that the matrix is in row-major order.
 */
void set(matrix *mat, int row, int col, double val) {
    // Task 1.1 TODO
    (mat -> data)[row*(mat -> cols) + col] = val;
}

/*
 * Allocates space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. `parent` should be set to NULL to indicate that
 * this matrix is not a slice. You should also set `ref_cnt` to 1.
 * You should return -1 if either `rows` or `cols` or both have invalid values. Return -2 if any
 * call to allocate memory in this function fails.
 * Return 0 upon success.
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    // Task 1.2 TODO
    if (rows <= 0 || cols <= 0) {
        return -1;
    }
    matrix* mat_ptr = (matrix*) malloc(sizeof(matrix));
    if (mat_ptr == NULL) {return -2;}
    mat_ptr->data = (double*) calloc(rows * cols, sizeof(double));
    if (mat_ptr->data == NULL) {return -2;}
    mat_ptr->rows = rows;
    mat_ptr->cols = cols;
    mat_ptr->parent = NULL;
    mat_ptr->ref_cnt = 1;
    *mat = (matrix*) mat_ptr;
    return 0;
    // HINTS: Follow these steps.
    // 1. Check if the dimensions are valid. Return -1 if either dimension is not positive.
    // 2. Allocate space for the new matrix struct. Return -2 if allocating memory failed.
    // 3. Allocate space for the matrix data, initializing all entries to be 0. Return -2 if allocating memory failed.
    // 4. Set the number of rows and columns in the matrix struct according to the arguments provided.
    // 5. Set the `parent` field to NULL, since this matrix was not created from a slice.
    // 6. Set the `ref_cnt` field to 1.
    // 7. Store the address of the allocated matrix struct at the location `mat` is pointing at.
    // 8. Return 0 upon success.
}

/*
 * You need to make sure that you only free `mat->data` if `mat` is not a slice and has no existing slices,
 * or that you free `mat->parent->data` if `mat` is the last existing slice of its parent matrix and its parent
 * matrix has no other references (including itself).
 */
void deallocate_matrix(matrix *mat) {
    // Task 1.3 TODO
    // HINTS: Follow these steps.
    // 1. If the matrix pointer `mat` is NULL, return.
    // 2. If `mat` has no parent: decrement its `ref_cnt` field by 1. If the `ref_cnt` field becomes 0, then free `mat` and its `data` field.
    // 3. Otherwise, recursively call `deallocate_matrix` on `mat`'s parent, then free `mat`.
    if(mat == NULL) {return;}
    if(mat->parent ==NULL) {
        mat -> ref_cnt -= 1;
        if (mat-> ref_cnt == 0) {
            free(mat->data);
            free(mat);
        }
    } else {
        deallocate_matrix(mat -> parent);
        free(mat);
    }
}

/*
 * Allocates space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * Its data should point to the `offset`th entry of `from`'s data (you do not need to allocate memory)
 * for the data field. `parent` should be set to `from` to indicate this matrix is a slice of `from`
 * and the reference counter for `from` should be incremented. Lastly, do not forget to set the
 * matrix's row and column values as well.
 * You should return -1 if either `rows` or `cols` or both have invalid values. Return -2 if any
 * call to allocate memory in this function fails.
 * Return 0 upon success.
 * NOTE: Here we're allocating a matrix struct that refers to already allocated data, so
 * there is no need to allocate space for matrix data.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int offset, int rows, int cols) {
    // Task 1.4 TODO
    if (rows <= 0 || cols <= 0) {
        return -1;
    }
    matrix* mat_ptr = (matrix*) malloc(sizeof(matrix));
    mat_ptr -> data = from -> data + offset;
    mat_ptr -> rows = rows;
    mat_ptr -> cols = cols;
    mat_ptr -> parent = from;
    from->ref_cnt += 1;
    *mat = (matrix*) mat_ptr;
    return 0;


    // HINTS: Follow these steps.
    // 1. Check if the dimensions are valid. Return -1 if either dimension is not positive.
    // 2. Allocate space for the new matrix struct. Return -2 if allocating memory failed.
    // 3. Set the `data` field of the new struct to be the `data` field of the `from` struct plus `offset`.
    // 4. Set the number of rows and columns in the new struct according to the arguments provided.
    // 5. Set the `parent` field of the new struct to the `from` struct pointer.
    // 6. Increment the `ref_cnt` field of the `from` struct by 1.
    // 7. Store the address of the allocated matrix struct at the location `mat` is pointing at.
    // 8. Return 0 upon success.
}

/*
 * Sets all entries in mat to val. Note that the matrix is in row-major order.
 */
void fill_matrix(matrix *mat, double val) {
    // Task 1.5 TODO

    // UNOPTIMIZED CODE
    // for(int i = 0; i< num_items; i = i + 1) {
    //     (mat->data)[i] = val;
    // }

    int num_items = mat->rows * mat->cols;
    int num_items_trunc = num_items / 4 * 4;
    __m256d vals = _mm256_set1_pd(val);

    #pragma omp parallel for
    for (int i = 0; i < num_items_trunc; i += 4) {
        _mm256_storeu_pd(mat -> data + i, vals);
    }

    for (int i = num_items_trunc; i < num_items; i += 1) {
        (mat -> data)[i] = val;
    }
//not doing x and y might cause error prob not. :)
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success.
 * Note that the matrix is in row-major order.
 */
int abs_matrix(matrix *result, matrix *mat) {
    // Task 1.5 TODO
    double* input = mat-> data; //*(mat->data);
    double* destination = result -> data;
    int num_items = mat->rows * mat->cols;
    int num_items_trunc = num_items / 4 * 4;

    __m256d minus_one = _mm256_set1_pd(-1);
    #pragma omp parallel for
    for(int i = 0 ; i < num_items_trunc; i += 4){
        __m256d curr_vec = _mm256_loadu_pd(input + i);
        __m256d abs_vec = _mm256_max_pd(_mm256_mul_pd(curr_vec, minus_one), curr_vec);
        _mm256_storeu_pd(destination + i, abs_vec);
    }

    for (int i = num_items_trunc; i < num_items; i += 1) {
        destination[i] = fabs(input[i]);
    }
    //UNOPTIMIZED
    // for(int i = 0; i< num_items; i = i + 1) {
    //     double curr = (mat->data)[i];
    //     (result->data)[i] = abs(curr);
    // }
    return 0;
}

/*
 * (OPTIONAL)
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success.
 * Note that the matrix is in row-major order.
 */
int neg_matrix(matrix *result, matrix *mat) {
    // Task 1.5 TODO
    return 0;
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success.
 * You may assume `mat1` and `mat2` have the same dimensions.
 * Note that the matrix is in row-major order.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    // Task 1.5 TODO
    //UNOPTIMIZED
    // int num_items = result->rows * result->cols;
    // for(int i = 0; i< num_items; i = i + 1) {
    //     double curr1 = (mat1->data)[i];
    //     double curr2 = (mat2->data)[i];
    //     (result->data)[i] = curr1 + curr2;
    // }

    double* mat1_input = mat1-> data; //*(mat->data);
    double* mat2_input = mat2-> data; //*(mat->data);
    double* destination = result -> data;
    int num_items = result->rows * result->cols;
    int num_items_trunc = num_items / 4 * 4;

    #pragma omp parallel for
    for(int i = 0 ; i < num_items_trunc; i += 4){
        __m256d add_vec = _mm256_add_pd(_mm256_loadu_pd(mat1_input + i), _mm256_loadu_pd(mat2_input + i));
        _mm256_storeu_pd(destination + i, add_vec);
    }

    for (int i = num_items_trunc; i < num_items; i += 1) {
        destination[i] = mat1_input[i] + mat2_input[i];
    }

    return 0;
}

/*
 * (OPTIONAL)
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success.
 * You may assume `mat1` and `mat2` have the same dimensions.
 * Note that the matrix is in row-major order.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    // Task 1.5 TODO
    return 0;
}

double dot_product(matrix* mat1, double* mat2_transpose, int r, int c) {
    // Returns the value of a dot product for row r of matrix 1 and col c of matrix 2
    double* mat1_ptr = mat1 -> data;
    int mat1_c = mat1->cols;

    __m256d dot_prod_vec = _mm256_set1_pd(0.0);
    for (int i=0; i < mat1_c/4*4; i+=4) {
        __m256d prod = _mm256_set1_pd(0.0);
        __m256d mat1_curr_vec =  _mm256_set_pd(mat1_ptr[r * mat1_c + i], mat1_ptr[r * mat1_c + i + 1],
                                               mat1_ptr[r * mat1_c + i + 2], mat1_ptr[r * mat1_c + i + 3]);
        __m256d matt2_curr_vec =  _mm256_set_pd(mat2_transpose[c * mat1_c + i], mat2_transpose[c * mat1_c + i + 1],
                                                mat2_transpose[c * mat1_c + i + 2], mat2_transpose[c * mat1_c + i + 3]);
        prod = _mm256_mul_pd(mat1_curr_vec, matt2_curr_vec);
        dot_prod_vec = _mm256_add_pd(dot_prod_vec, prod);
    }
    double prod_arr[4];
    _mm256_storeu_pd(prod_arr, dot_prod_vec);
    double dot_prod = prod_arr[0] + prod_arr[1] + prod_arr[2] + prod_arr[3];

    // Tail case
    for(int i = mat1_c/4*4; i < mat1_c; i ++) {
        dot_prod += mat1_ptr[r * mat1_c + i] * mat2_transpose[c * mat1_c + i];
    }

    return dot_prod;
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 * You may assume `mat1`'s number of columns is equal to `mat2`'s number of rows.
 * Note that the matrix is in row-major order.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    // Transpose matrix 2
    int mat2_r = mat2->rows;
    int mat2_c = mat2->cols;
    double* mat2_ptr = mat2 -> data;

    int mat1_c = mat1->cols;
    double* mat1_ptr = mat1 -> data;

    // Eli's better transpose
    double* mat2_transpose = (double*) malloc(mat2_r * mat2_c * sizeof(double));
    #pragma omp parallel for
    for (int r=0; r< mat2_r; r ++) {
        for (int c=0; c< mat2_c; c ++) {
            mat2_transpose[c * mat2_r + r] = mat2_ptr[r*mat2_c + c];
        }
    }

    //MARTRIX MULT
    #pragma omp parallel for
    for (int r=0; r<result->rows; r++) {
        for (int c=0; c<result->cols; c++) {
            __m256d dot_prod_vec = _mm256_set1_pd(0.0);
            for (int i=0; i < mat1_c/32*32; i+=32) {
                // prod 1
                __m256d mat1_curr_vec = _mm256_loadu_pd(mat1_ptr + r * mat1_c + i);
                __m256d mat2_curr_vec = _mm256_loadu_pd(mat2_transpose+ c * mat1_c + i);
                dot_prod_vec = _mm256_add_pd(dot_prod_vec, _mm256_mul_pd(mat1_curr_vec, mat2_curr_vec));
                // prod 2
                mat1_curr_vec = _mm256_loadu_pd(mat1_ptr + r * mat1_c + i + 4);
                mat2_curr_vec = _mm256_loadu_pd(mat2_transpose+ c * mat1_c + i + 4);
                dot_prod_vec = _mm256_add_pd(dot_prod_vec, _mm256_mul_pd(mat1_curr_vec, mat2_curr_vec));
                // prod 3
                mat1_curr_vec = _mm256_loadu_pd(mat1_ptr + r * mat1_c + i + 8);
                mat2_curr_vec = _mm256_loadu_pd(mat2_transpose+ c * mat1_c + i + 8);
                dot_prod_vec = _mm256_add_pd(dot_prod_vec, _mm256_mul_pd(mat1_curr_vec, mat2_curr_vec));
                // prod 4
                mat1_curr_vec = _mm256_loadu_pd(mat1_ptr + r * mat1_c + i + 12);
                mat2_curr_vec = _mm256_loadu_pd(mat2_transpose+ c * mat1_c + i + 12);
                dot_prod_vec = _mm256_add_pd(dot_prod_vec, _mm256_mul_pd(mat1_curr_vec, mat2_curr_vec));
                // prod 5
                mat1_curr_vec = _mm256_loadu_pd(mat1_ptr + r * mat1_c + i + 16);
                mat2_curr_vec = _mm256_loadu_pd(mat2_transpose+ c * mat1_c + i + 16);
                dot_prod_vec = _mm256_add_pd(dot_prod_vec, _mm256_mul_pd(mat1_curr_vec, mat2_curr_vec));
                // prod 6
                mat1_curr_vec = _mm256_loadu_pd(mat1_ptr + r * mat1_c + i + 20);
                mat2_curr_vec = _mm256_loadu_pd(mat2_transpose+ c * mat1_c + i + 20);
                dot_prod_vec = _mm256_add_pd(dot_prod_vec, _mm256_mul_pd(mat1_curr_vec, mat2_curr_vec));
                // prod 7
                mat1_curr_vec = _mm256_loadu_pd(mat1_ptr + r * mat1_c + i + 24);
                mat2_curr_vec = _mm256_loadu_pd(mat2_transpose+ c * mat1_c + i + 24);
                dot_prod_vec = _mm256_add_pd(dot_prod_vec, _mm256_mul_pd(mat1_curr_vec, mat2_curr_vec));
                // prod 8
                mat1_curr_vec = _mm256_loadu_pd(mat1_ptr + r * mat1_c + i + 28);
                mat2_curr_vec = _mm256_loadu_pd(mat2_transpose+ c * mat1_c + i + 28);
                dot_prod_vec = _mm256_add_pd(dot_prod_vec, _mm256_mul_pd(mat1_curr_vec, mat2_curr_vec));
            }
            double prod_arr[4];
            _mm256_storeu_pd(prod_arr, dot_prod_vec);
            double dot_prod = prod_arr[0] + prod_arr[1] + prod_arr[2] + prod_arr[3];

            // Tail case
            for(int i = mat1_c/32*32; i < mat1_c; i ++) {
                dot_prod += mat1_ptr[r * mat1_c + i] * mat2_transpose[c * mat1_c + i];
            }

            (result->data)[r*result->cols + c] = dot_prod;
        }
    }

    free(mat2_transpose);
    return 0;
}

void copy_matrix_from(matrix* from, matrix* to) {
    fill_matrix(to, 0);
    for(int r=0; r<from->rows; r++) {
        for(int c=0; c<from->cols; c++) {
            set(to, r, c, get(from, r, c));
        }
    }
}

int make_identity(matrix* mat) {
    fill_matrix(mat, 0);
    int size = mat-> cols;
    #pragma omp parallel for
    for (int i = 0; i < size/4*4; i += 4) {
        mat ->data[i *size + i] = 1;
        mat ->data[(i+1) *size + (i+1)] = 1;
        mat ->data[(i+2) *size + (i+2)] = 1;
        mat ->data[(i+3) *size + (i+3)] = 1;
    }

    //tail case
    for (int i = size/4*4; i < size; i ++ ) {
        mat->data[i*size + i] = 1;
    }
    return 0;
}

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 * You may assume `mat` is a square matrix and `pow` is a non-negative integer.
 * Note that the matrix is in row-major order.
 */
int pow_matrix(matrix *result, matrix *mat, int pow) {
    int size = mat -> cols;
    if (pow == 0){
        make_identity(result);
        return 0;
    }

    //copy contents into result
    if(pow == 1) {
        memcpy(result -> data, mat -> data, sizeof(double) * size * size);
        return 0;
    }

    // set MX to mat
    matrix* MX = NULL;
    allocate_matrix(&MX, size, size);
    memcpy(MX -> data, mat -> data, sizeof(double) * size * size);

    // set MY to identity
    matrix* MY = NULL;
    allocate_matrix(&MY, size, size);
    make_identity(MY);
    while(pow > 1){
        if(pow % 2 == 0) {
            mul_matrix(result, MX, MX);
            memcpy(MX -> data, result -> data, sizeof(double) * size * size);
            pow = pow/2;
        } else {
            mul_matrix(result, MY, MX);
            memcpy(MY -> data, result -> data, sizeof(double) * size * size);
            mul_matrix(result, MX, MX);
            memcpy(MX -> data, result -> data, sizeof(double) * size * size);
            pow = (pow-1)/2;
        }
    }

    mul_matrix(result, MX, MY);

    return 0;
}


//valgrind --leak-check=full -v ./matrix.o
// https://www.gradescope.com/courses/343452/assignments/1995037/submissions/123549663

// UV's transpose
// double *mat2_transpose = (double*) malloc( mat2_r * mat2_c * sizeof(double));
    // if (mat2_transpose == NULL) {
    //     return -1;
    // }
    // int shift = 0;
    // for (int i = 0; i < mat2_c; i++){
    //     for (int j=0; j < mat2_r/4*4; j += 4) {
    //         __m256d val = _mm256_set_pd((mat2->data)[(j+3)*mat2_c + i], (mat2->data)[(j+2)*mat2_c + i],
    //                                     (mat2->data)[(j+1)*mat2_c + i], (mat2->data)[(j)*mat2_c + i]);
    //         __m256d val = _mm256_loadu_pd()
    //         _mm256_storeu_pd(mat2_transpose + shift, val);
    //         shift += 4;
    //     }

    //     // Tail case
    //     for (int j = mat2_r/4*4; j <mat2_r; j++) {
    //         mat2_transpose[shift] = mat2_ptr[j*mat2_c +i];
    //         shift ++;
    //     }
    // }
