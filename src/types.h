/*
 * types.h - Vector Cache Database Implementation
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

#ifndef _VICTOR_TYPES_H
#define _VICTOR_TYPES_H 1

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define STORE_SIZE (1024*1024)
#define ALIGN_DIMS(d) (((d) + 3) & ~3)
#define MAX_BUCKETS 128

#define SVEC_SIZE(d) (STORE_SIZE / ((d) * sizeof(float32_t))) 

typedef float float32_t;

struct bucket {
    int index;
    float32_t **svec;
    float32_t *store;
};

struct table {
    int dims;
    int dims_aligned;
    int svec_size;

    pthread_rwlock_t rwlock;

    int cmpmode;
    float32_t worst_match_value;

    int index;

    int       (*is_better_match) (float32_t,   float32_t);
    float32_t (*compare_vectors) (float32_t *, float32_t *, int);
    
    struct bucket *buckets[MAX_BUCKETS];
};

typedef struct table table_t;

#define L2NORM 0x01
#define COSINE 0x02

typedef struct {
    int id;
    float32_t distance;
} match_result_t;

#endif /* VICTOR_TYPES */
