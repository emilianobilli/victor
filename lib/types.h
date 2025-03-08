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

#ifndef __TYPES_H
#define __TYPES_H 1

#include <stdint.h>
#include <stdlib.h>

#define ALIGN_DIMS(d) (((d) + 3) & ~3)

typedef float float32_t;

typedef struct {
    int id;
    float32_t distance;
} MatchResult;


typedef enum {
    SUCCESS,
    INVALID_INIT,
    INVALID_INDEX,
    INVALID_VECTOR,
    INVALID_RESULT,
    INVALID_DIMENSIONS,
    INVALID_ID,
    INDEX_EMPTY,
    SYSTEM_ERROR,
} ErrorCode;

#endif /* TYPES */
