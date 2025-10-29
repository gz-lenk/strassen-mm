#ifndef _BLOCK_MM_H 
#define _BLOCK_MM_H

#include "hls_stream.h"
#include <iostream>
#include <iomanip>
#include <vector>


using namespace std;

#define M 24
#define K 24
#define N 24

#define M_factor 4
#define K_factor 4
#define N_factor 4

// TODO: Change the code to allow it to handle matrices of arbitrary size.
typedef int8_t DTYPE;
#define BLOCK_SIZE 24
#define TILE_SIZE 6

typedef struct{
    DTYPE a[TILE_SIZE];
} tilevec;

typedef struct{
    DTYPE out[TILE_SIZE][TILE_SIZE];
} tilemat;

void tile_matmul_part(
    hls::stream<tilevec>& Arows,
    hls::stream<tilevec>& Bcols,
    tilemat& ABpart,
    DTYPE iteration
);



#endif