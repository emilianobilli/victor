/*
 * victor.h - Vector Cache Database Implementation
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

#ifndef _VICTOR_H
#define _VICTOR_H

#include "types.h"

extern struct table *victor_table(int dims, int cmpmode);
extern int insert_vector(struct table *table, float32_t *vector);
extern int delete_vector(struct table *table, int id);
extern match_result_t search_better_match(struct table *table, float32_t *vector);
extern void free_table(struct table **table);
#endif