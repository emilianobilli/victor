/*
 * victor.c - Vector Cache Database Implementation
 * 
 * Copyright (C) 2025 Emiliano A. Billi
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 *
 * Contact: emiliano.billi@gmail.com
 */

#include "types.h"
#include "math.h"


/**
 * @brief Allocates memory for a new bucket to store vectors.
 *
 * This function creates and initializes a new bucket, which consists of a 
 * contiguous memory block (`store`) for vector storage and an array of 
 * pointers (`svec`) that reference positions within `store`. The bucket 
 * is designed to hold a fixed number of vectors, ensuring memory alignment 
 * for optimal SIMD performance.
 *
 * @param dims_aligned The number of dimensions, aligned to a multiple of 4.
 * @return A pointer to the newly allocated `struct bucket`, or `NULL` if allocation fails.
 *
 * @note The allocated memory for vectors (`store`) is zero-initialized.
 * @note The `svec` array holds pointers to each vector stored in `store`.
 * @note If any allocation fails, the function ensures proper cleanup before returning `NULL`.
 */

static struct bucket *alloc_bucket(int dims_aligned) {
    struct bucket *bucket = (struct bucket *) malloc(sizeof(struct bucket));
    int i, size = SVEC_SIZE(dims_aligned);
    if (bucket == NULL) 
        return NULL;

    bucket->store = (float32_t *) calloc(1, STORE_SIZE);
    if (bucket->store == NULL) {
        free(bucket);
        return NULL;
    }

    bucket->svec = (float32_t **) calloc(size, sizeof(float32_t *));
    if (bucket->svec == NULL) {
        free(bucket->store);
        free(bucket);
        return NULL;
    }
    bucket->index = 0;
    for (i=0; i < size; i++) {
        bucket->svec[i] = bucket->store + (i * dims_aligned);
    }
    return bucket;
}

/**
 * @brief Frees the memory allocated for a bucket.
 *
 * This function deallocates all memory associated with a given bucket, 
 * including its vector storage (`store`) and pointer array (`svec`). 
 * It also sets the bucket pointer to `NULL` to prevent dangling references.
 *
 * @param bucket_ptr A double pointer to the `struct bucket` to be freed.
 *
 * @note If `bucket_ptr` is `NULL` or already freed, the function does nothing.
 * @note After calling this function, `*bucket_ptr` will be set to `NULL`.
 */
static void free_bucket(struct bucket **bucket_ptr) {
    if (bucket_ptr == NULL || *bucket_ptr == NULL) return;  

    struct bucket *bucket = *bucket_ptr;  


    if (bucket->store != NULL) {
        free(bucket->store);
        bucket->store = NULL;
    }
    if (bucket->svec != NULL) {
        free(bucket->svec);
        bucket->svec = NULL;
    }

    free(bucket);
    *bucket_ptr = NULL;
}


/**
 * @brief Generates a unique 32-bit ID from a bucket index and a slot index.
 *
 * This function encodes the bucket index and slot index into a single 
 * 32-bit integer. The bucket index occupies the higher 8 bits (signed), 
 * while the slot index occupies the lower 24 bits (unsigned). This ensures 
 * that each vector has a unique identifier within the table.
 *
 * @param bucket The bucket index (signed 8-bit integer).
 * @param slot The slot index within the bucket (unsigned 32-bit integer).
 * @return A unique 32-bit ID combining the bucket and slot indices.
 *
 * @note The bucket index is stored in the most significant 8 bits of the ID.
 * @note The slot index is stored in the least significant 24 bits.
 * @note The function is `static inline` for efficient inlining at compile time.
 */
static inline int32_t encode_vector_id(int8_t bucket, uint32_t slot) {
    return ((int32_t)bucket << 24) | (slot & 0x00FFFFFF);
}


/**
 * @brief Finds the most similar vector in the table without a threshold constraint.
 *
 * This function searches for the most similar vector to the given query `vector`
 * within all stored vectors in the `table`. The similarity measure is determined
 * by the comparison mode (`cmpmode`) set in the table structure.
 *
 * Unlike `victor_cmpvec_th`, this function does not perform early stopping based 
 * on a threshold. Instead, it always returns the closest match found in the table.
 *
 * @param table Pointer to the table containing stored vectors.
 * @param vector Pointer to the query vector (array of float32_t).
 * @return A `victor_retval_t` structure containing:
 *         - `id`: The ID of the most similar vector found (`-1` if no vectors exist).
 *         - `val`: The computed similarity or distance value.
 *
 * @note The function uses the `cmpvec` function pointer inside `table`
 *       to determine similarity (e.g., Euclidean distance or cosine similarity).
 * @note If no vectors are stored in the table, the function returns `id = -1`
 *       and a default value for `val` (e.g., `INFINITY` for distances).
 */
match_result_t search_better_match(struct table *table, float32_t *vector) {
    match_result_t result;
    struct bucket *b;
    float32_t tmp;
    int i, j;

    pthread_rwlock_rdlock(&table->rwlock);

    result.distance = table->worst_match_value;
    result.id = -1;
    


    for (i = 0; i <= table->index; i++) {
        b = table->buckets[i];
        if (b == NULL) continue;
        
        for (j = 0; j < b->index; j++) {
            if (b->svec[j] == NULL) continue;
            
            tmp = table->compare_vectors(b->svec[j], vector, table->dims_aligned);
            if (table->is_better_match(tmp, result.distance)) {
                result.id = encode_vector_id(i, j);
                result.distance = tmp;
            }
        }
    }
    pthread_rwlock_unlock(&table->rwlock);
    return result;
}

static void shift_right_mr(match_result_t *result, int len) {
    int i;
    for (i = len-1; i > 0; i--) {
        result[i].id = result[i-1].id;
        result[i].distance = result[i-1].distance;
    }
    return;
}

/**
 * @brief Searches for the n best matching vectors in the table.
 *
 * This function iterates over all the buckets and vectors stored in the given table,
 * comparing each vector with the query vector using the table's compare_vectors function.
 * It maintains an array of n best match results (each of type match_result_t) sorted by 
 * increasing distance (or, equivalently, by better match criteria as defined in is_better_match).
 * When a new candidate vector is found that is better than one of the current n best matches,
 * the function shifts the existing matches to make room for the new candidate using shift_right_mr.
 *
 * @param table Pointer to the table structure containing stored vectors, buckets, and the comparison functions.
 * @param vector Pointer to the query vector (array of float32_t) for which matches are sought.
 * @param result Address of a pointer where the function will allocate an array of n match_result_t structures 
 *               representing the n best matches.
 * @param n The number of best match results to return.
 * @return 0 on success, or -1 if memory allocation fails.
 *
 * @note The function initializes the result array with a default worst_match_value and an id of -1 for each entry.
 * @note It uses table->compare_vectors to compute a distance (or similarity) between vectors, and table->is_better_match 
 *       to decide whether a new candidate is a better match compared to an existing result.
 * @note The new candidate is inserted in the result array at the appropriate position after shifting the elements 
 *       (via shift_right_mr) starting from that position.
 * @note The allocated result array must be freed by the caller when no longer needed.
 */
int search_better_n_match(struct table *table, float32_t *vector, match_result_t **result, int n) {
    struct bucket *b;
    float32_t tmp;
    int i, j, k;
    *result = malloc(n * sizeof(match_result_t));
    if (*result == NULL) return -1;
    
    pthread_rwlock_rdlock(&table->rwlock);
    for (i = 0; i <n; i++) {
        (*result)[i].distance = table->worst_match_value;
        (*result)[i].id = -1;
    }


    for (i = 0; i <= table->index; i++) {
        b = table->buckets[i];
        if (b == NULL) continue;
        
        for (j = 0; j < b->index; j++) {
            if (b->svec[j] == NULL) continue;
            
            tmp = table->compare_vectors(b->svec[j], vector, table->dims_aligned);

            for (k = 0; k < n; k++) {
                if (table->is_better_match(tmp, (*result)[k].distance)) {
                    shift_right_mr(&(*result)[k], n-k-1);
                    (*result)[k].distance = tmp;
                    (*result)[k].id = encode_vector_id(i, j);
                    break;
                }
            }
        }
    }
    pthread_rwlock_unlock(&table->rwlock);
    return 0;
}

/**
 * @brief Inserts a vector into the table.
 *
 * This function stores a new vector in the table, allocating additional 
 * memory if necessary. If the current bucket is full, a new bucket is 
 * created to continue storing vectors.
 *
 * @param table Pointer to the table where the vector will be inserted.
 * @param vector Pointer to the vector (array of float32_t) to be inserted.
 * @return The unique ID assigned to the inserted vector, or `-1` if insertion fails.
 *
 * @note The table maintains a linked list of buckets, each containing multiple vectors.
 * @note If there is no available space in the current bucket, a new bucket is allocated.
 * @note The ID format includes both the bucket index and the vector's position within it.
 * @note The function does not check for duplicate vectors.
 */
int insert_vector(struct table *table, float32_t *vector) {
    pthread_rwlock_wrlock(&table->rwlock);
    struct bucket *b = table->buckets[table->index];
    int id;
    if (b->index >= table->svec_size) {
        if (table->index + 1 < MAX_BUCKETS) {
            b = alloc_bucket(table->dims_aligned);
            if (b == NULL) {
                pthread_rwlock_unlock(&table->rwlock);
                return -1;
            }
            table->index++;
            table->buckets[table->index] = b;
        } else {
            pthread_rwlock_unlock(&table->rwlock);
            return -1;
        }
    }
    memcpy(b->svec[b->index], vector, (table->dims * sizeof(float32_t)));
    memset(b->svec[b->index] + table->dims, 0, (table->dims_aligned - table->dims) * sizeof(float32_t));
    id = (int) encode_vector_id(table->index, b->index);
    b->index++;
    pthread_rwlock_unlock(&table->rwlock);
    return id;
}

/**
 * @brief Deletes a vector from the table.
 *
 * This function removes a vector from the table based on its unique ID.
 * The vector's memory space is cleared, but the ID slot is not reused,
 * ensuring that deleted vectors do not interfere with future insertions.
 *
 * @param table Pointer to the table containing the vector.
 * @param id The unique ID of the vector to be deleted.
 * @return `0` if the deletion is successful, `-1` if the ID is invalid or the vector does not exist.
 *
 * @note The function does not compact the storage or shift existing vectors.
 * @note The slot remains empty after deletion and is not reused.
 * @note The ID is structured with the bucket index in the higher bits and the slot index in the lower bits.
 * @note If the ID is out of bounds or references a non-existent vector, the function returns `-1`.
 */
int delete_vector(struct table *table, int id) {
    pthread_rwlock_wrlock(&table->rwlock);
    int ib = (int8_t)(id >> 24); 
    int iv = id & 0x00FFFFFF;

    if (ib >= 0 && ib < MAX_BUCKETS && table->buckets[ib] != NULL && iv >=0 && iv < table->svec_size && table->buckets[ib]->svec[iv] != NULL) {
        struct bucket *b = table->buckets[ib];
        memset(b->svec[iv], 0, table->dims_aligned  * sizeof(float32_t));
        b->svec[iv] = NULL;
    }
    pthread_rwlock_unlock(&table->rwlock);
    return 0;
}

/**
 * @brief Creates and initializes a new vector table.
 *
 * This function allocates memory for a new table structure and initializes 
 * its parameters, including dimensionality and comparison mode. The table 
 * starts with a single allocated bucket for storing vectors.
 *
 * @param dims The number of dimensions per vector.
 * @param cmpmode The comparison mode used for similarity calculations 
 *        (e.g., `L2NORM` for Euclidean distance or `COSINE` for cosine similarity).
 * @return A pointer to the newly created table, or `NULL` if allocation fails.
 *
 * @note The table maintains a linked list of buckets for dynamic vector storage.
 * @note The function aligns `dims` to a multiple of 4 to optimize SIMD operations.
 * @note The first bucket is allocated immediately upon table creation.
 * @note If an invalid `cmpmode` is provided, the function returns `NULL`.
 */
struct table *victor_table(int dims, int cmpmode) {
    struct table *t = (struct table *) calloc(1, sizeof(struct table));
    if (t == NULL) return NULL;  // Fallo en la asignación de memoria


    t->dims = dims;
    t->dims_aligned = ALIGN_DIMS(dims);  // Asegurar alineación a múltiplo de 4
    t->svec_size = SVEC_SIZE(t->dims_aligned);

    pthread_rwlock_init(&t->rwlock, NULL);
    t->cmpmode = cmpmode;
    switch (cmpmode) {
        case L2NORM:
            t->compare_vectors = euclidean_distance;
            t->is_better_match = euclidean_distance_best;
            t->worst_match_value = INFINITY;
            break;
        case COSINE:
            t->compare_vectors = cosine_similarity;
            t->is_better_match = cosine_similarity_best;
            t->worst_match_value = -1.0f;
            break;
        default:
            free(t);
            return NULL;
    }
    
    t->index = 0;
    t->buckets[t->index] = alloc_bucket(t->dims_aligned);
    if (!(t->buckets[t->index])) {
        free(t);
        return NULL;
    }
    return t;
}

/**
 * @brief Frees the memory allocated for a vector table and its associated buckets.
 *
 * This function deallocates all memory associated with the given `table` structure, 
 * including its buckets. It ensures that no memory leaks occur and sets the table 
 * pointer to `NULL` after deallocation.
 *
 * @param table A double pointer to the `struct table` to be freed.
 *
 * @note This function safely handles NULL pointers to avoid segmentation faults.
 * @note Assumes `MAX_BUCKETS` defines the maximum number of buckets allocated.
 *
 * @warning If the `buckets` array was dynamically allocated, it should be freed separately 
 *          before calling this function to prevent memory leaks.
 */
void free_table(struct table **table){
    int i;
    if (table == NULL || *table == NULL)
        return;
    for (i = 0; i < MAX_BUCKETS; i++) 
        if ((*table)->buckets[i])
            free_bucket(&(*table)->buckets[i]);
    pthread_rwlock_destroy(&(*table)->rwlock);
    free(*table);
    *table = NULL;
}
