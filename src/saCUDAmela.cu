/*
 * Jhon Steeven Cabanilla Alvarado
 * Miguel chaveinte García
 * G01
 * 
 * Probabilistic approach to locate maximum heights
 * Hill Climbing + Montecarlo
 *
 * CUDA version
 *
 * Computacion Paralela, Grado en Informatica (Universidad de Valladolid)
 * 2021/2022
 *
 * v1.1
 *
 * (c) 2022 Arturo Gonzalez Escribano
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <sys/time.h>

/* Headers for the CUDA assignment versions */
#include <cuda.h>

/*
 * Macros to show errors when calling a CUDA library function,
 * or after launching a kernel
 */
#define CHECK_CUDA_CALL(a)                                                                            \
	{                                                                                                 \
		cudaError_t ok = a;                                                                           \
		if (ok != cudaSuccess)                                                                        \
			fprintf(stderr, "-- Error CUDA call in line %d: %s\n", __LINE__, cudaGetErrorString(ok)); \
	}
#define CHECK_CUDA_LAST()                                                                             \
	{                                                                                                 \
		cudaError_t ok = cudaGetLastError();                                                          \
		if (ok != cudaSuccess)                                                                        \
			fprintf(stderr, "-- Error CUDA last in line %d: %s\n", __LINE__, cudaGetErrorString(ok)); \
	}

#define PRECISION 10000

/*
 * Structure to represent a climbing searcher
 * 	This structure can be changed and/or optimized by the students
 */
typedef struct
{
	int pos_row, pos_col; // Position in the grid
	int steps;			  // Steps count
	int follows;		  // When it finds an explored trail, who searched that trail
} Searcher;

/*
 * Function to get wall time
 */
double cp_Wtime()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + 1.0e-6 * tv.tv_usec;
}

/*
 * Macro function to simplify accessing with two coordinates to a flattened array
 * 	This macro-function can be changed and/or optimized by the students
 */
#define accessMat(arr, exp1, exp2) arr[(int)(exp1)*columns + (int)(exp2)]

/*
 * Function: Generate height for a given position
 * 	This function can be changed and/or optimized by the students
 */

//***********************************MODIFICACION **************************************************************

__global__ void reductionMax(int *array, int size, int *result)
{
	// Compute the global position of the thread in the grid
	int globalPos = threadIdx.x + blockIdx.x * blockDim.x;

	// Shared memory: One element per thread in the block
	// Call this kernel with the proper third launching parameter
	extern __shared__ int buffer[];

	// Load array values in the shared memory (0 if out of the array)
	if (globalPos < size)
	{
		buffer[threadIdx.x] = array[globalPos];
	}
	else
		buffer[threadIdx.x] = 0;

	// Wait for all the threads of the block to finish
	__syncthreads();

	// Reduction tree
	for (int step = blockDim.x / 2; step >= 1; step /= 2)
	{
		if (threadIdx.x < step)
			if (buffer[threadIdx.x] < buffer[threadIdx.x + step])
				buffer[threadIdx.x] = buffer[threadIdx.x + step];
		__syncthreads();
	}

	// The maximum value of this block is on the first position of buffer
	if (threadIdx.x == 0)
		atomicMax(result, buffer[0]);
}

__global__ void reductionAdd(int *array, int size, int *result)
{
	// Compute the global position of the thread in the grid
	int globalPos = threadIdx.x + blockIdx.x * blockDim.x;

	// Shared memory: One element per thread in the block
	// Call this kernel with the proper third launching parameter
	extern __shared__ int buffer[];

	// Load array values in the shared memory (0 if out of the array)
	if (globalPos < size)
	{ // and array[globalPos] != INT_MIN
		if (array[globalPos] != -1)
		{
			buffer[threadIdx.x] = 1;
		}
		else
		{
			buffer[threadIdx.x] = 0;
		}
	}
	else
		buffer[threadIdx.x] = 0;

	// Wait for all the threads of the block to finish
	__syncthreads();

	// Reduction tree
	for (int step = blockDim.x / 2; step >= 1; step /= 2)
	{
		if (threadIdx.x < step)
			buffer[threadIdx.x] += buffer[threadIdx.x + step];
		__syncthreads();
	}

	// The maximum value of this block is on the first position of buffer
	if (threadIdx.x == 0)
		atomicAdd(result, buffer[0]);
}

__global__ void reductionAddHeights(int *array, int size, unsigned long long int *result)
{
	// Compute the global position of the thread in the grid
	int globalPos = threadIdx.x + blockIdx.x * blockDim.x;

	// Shared memory: One element per thread in the block
	// Call this kernel with the proper third launching parameter
	extern __shared__ int buffer[];

	// Load array values in the shared memory (0 if out of the array)
	if (globalPos < size)
	{ // and array[globalPos] != INT_MIN
		if (array[globalPos] != INT_MIN)
		{
			buffer[threadIdx.x] = array[globalPos];
		}
		else
		{
			buffer[threadIdx.x] = 0;
		}
	}
	else
		buffer[threadIdx.x] = 0;

	// Wait for all the threads of the block to finish
	__syncthreads();

	// Reduction tree
	for (int step = blockDim.x / 2; step >= 1; step /= 2)
	{
		if (threadIdx.x < step)
			buffer[threadIdx.x] += buffer[threadIdx.x + step];
		__syncthreads();
	}

	// The maximum value of this block is on the first position of buffer
	if (threadIdx.x == 0)
		atomicAdd(result, buffer[0]);
}

__device__ int get_height(int x, int y, int rows, int columns, float x_min, float x_max, float y_min, float y_max)
{
	/* Calculate the coordinates of the point in the ranges */
	float x_coord = x_min + ((x_max - x_min) / rows) * x;
	float y_coord = y_min + ((y_max - y_min) / columns) * y;
	/* Compute function value */
	float value = 2 * sin(x_coord) * cos(y_coord / 2) + log(fabs(y_coord - M_PI_2));
	/* Transform to fixed point precision */
	int fixed_point = (int)(PRECISION * value);
	return fixed_point;
}

/*
 * Function: Climbing step
 * 	This function can be changed and/or optimized by the students
 */

__device__ int climbing_step(int rows, int columns, Searcher *searchers, int search, int *heights, int *trails, float x_min, float x_max, float y_min, float y_max)
{

	int search_flag = 0;

	/* Annotate one step more, landing counts as the first step */
	searchers[search].steps++;

	/* Get starting position */
	int pos_row = searchers[search].pos_row;
	int pos_col = searchers[search].pos_col;

	/* Stop if searcher finds another trail */
	int check;
	// check = atomicAdd(&accessMat( tainted, pos_row, pos_col ), 1);
	check = atomicCAS(&accessMat(trails, pos_row, pos_col), -1, search);

	if (check != -1)
	{
		searchers[search].follows = check;
		search_flag = 1;
	}
	else
	{
		/* Annotate the trail */
		// accessMat(trails, pos_row, pos_col) = search; // Escritura

		/* Compute the height */
		accessMat(heights, pos_row, pos_col) = get_height(pos_row, pos_col, rows, columns, x_min, x_max, y_min, y_max);

		/* Locate the highest climbing direction */
		float local_max = accessMat(heights, pos_row, pos_col);
		int climbing_direction = 0;
		float altura;

		if (pos_row > 0)
		{
			/* Compute the height in the neighbor if needed */
			altura = accessMat(heights, pos_row - 1, pos_col);
			if (altura == INT_MIN)
				altura = accessMat(heights, pos_row - 1, pos_col) = get_height(pos_row - 1, pos_col, rows, columns, x_min, x_max, y_min, y_max);

			/* Annotate the travelling direction if higher */
			if (altura > local_max)
			{
				climbing_direction = 1;
				local_max = altura;
			}
		}

		if (pos_row < rows - 1)
		{
			/* Compute the height in the neighbor if needed */
			altura = accessMat(heights, pos_row + 1, pos_col);
			if (altura == INT_MIN)
				altura = accessMat(heights, pos_row + 1, pos_col) = get_height(pos_row + 1, pos_col, rows, columns, x_min, x_max, y_min, y_max);

			/* Annotate the travelling direction if higher */
			if (altura > local_max)
			{
				climbing_direction = 2;
				local_max = altura;
			}
		}

		if (pos_col > 0)
		{
			/* Compute the height in the neighbor if needed */
			altura = accessMat(heights, pos_row, pos_col - 1);
			if (altura == INT_MIN)
				altura = accessMat(heights, pos_row, pos_col - 1) = get_height(pos_row, pos_col - 1, rows, columns, x_min, x_max, y_min, y_max);

			/* Annotate the travelling direction if higher */
			if (altura > local_max)
			{
				climbing_direction = 3;
				local_max = altura;
			}
		}

		if (pos_col < columns - 1)
		{
			/* Compute the height in the neighbor if needed */
			altura = accessMat(heights, pos_row, pos_col + 1);
			if (altura == INT_MIN)
				altura = accessMat(heights, pos_row, pos_col + 1) = get_height(pos_row, pos_col + 1, rows, columns, x_min, x_max, y_min, y_max);

			/* Annotate the travelling direction if higher */
			if (altura > local_max)
			{
				climbing_direction = 4;
				local_max = altura;
			}
		}

		/* Stop if local maximum is reached */
		if (climbing_direction == 0)
		{
			searchers[search].follows = search;
			search_flag = 1;
		}
		else if (climbing_direction == 1)
			searchers[search].pos_row = --pos_row;
		else if (climbing_direction == 2)
			searchers[search].pos_row = ++pos_row;
		else if (climbing_direction == 3)
			searchers[search].pos_col = --pos_col;
		else
			searchers[search].pos_col = ++pos_col;
	}

	/* Return a flag to indicate if search should stop */
	return search_flag;
}

__global__ void searchersInitialization(Searcher *searchers, int *total_steps, int num_searchers)
{

	// Calculamos la posicion global de cada hilo
	int gid = threadIdx.x + blockDim.x * blockIdx.x;

	// Podemos tener más hilos de los necesarios. Por ello, añadimos este IF para que los hilos ociosos no entren en esta parte del codigo
	if (gid < num_searchers)
	{
		searchers[gid].steps = 0;
		searchers[gid].follows = -1;
		total_steps[gid] = 0;
	}
}

__global__ void computeFollower(Searcher *searchers, int num_searchers)
{

	// Calculamos la posicion global de cada hilo
	int gid = threadIdx.x + blockDim.x * blockIdx.x;

	if (gid < num_searchers)
	{
		int search_flag = 0;
		int parent = gid; // antes search
		int follows_to = searchers[parent].follows;
		while (!search_flag)
		{
			if (follows_to == parent)
				search_flag = 1;
			else
			{
				parent = follows_to;
				follows_to = searchers[parent].follows;
			}
		}
		searchers[gid].follows = follows_to;
	}
}

__global__ void computeAccumulatedTrail(Searcher *searchers, int *total_steps, int num_searchers)
{

	// Calculamos la posicion global de cada hilo
	int gid = threadIdx.x + blockDim.x * blockIdx.x;

	if (gid < num_searchers)
	{
		int pos_max = searchers[gid].follows;
		// total_steps[ pos_max ] += searchers[ gid ].steps;
		atomicAdd(&total_steps[pos_max], searchers[gid].steps);
	}
}

__global__ void computeSearchersTrails(int num_searchers, int rows, int columns, Searcher *searchers, int *heights, int *trails, float x_min, float x_max, float y_min, float y_max)
{

	// Calculamos la posicion global de cada hilo
	int gid = threadIdx.x + blockDim.x * blockIdx.x;

	int search_flag = 0;

	if (gid < num_searchers)
	{
		while (!search_flag)
		{
			search_flag = climbing_step(rows, columns, searchers, gid, heights, trails, x_min, x_max, y_min, y_max);
		}
	}
}

__global__ void computeFollows(Searcher *searchers, int *trails, int num_searchers, int columns)
{

	// Calculamos la posicion global de cada hilo
	int gid = threadIdx.x + blockDim.x * blockIdx.x;

	// Podemos tener más hilos de los necesarios. Por ello, añadimos este IF para que los hilos ociosos no entren en esta parte del codigo
	if (gid < num_searchers)
	{
		int pos_row = searchers[gid].pos_row;
		int pos_col = searchers[gid].pos_col;

		searchers[gid].follows = accessMat(trails, pos_row, pos_col);
	}
}

__global__ void computeLocalMax(Searcher *array, int size, int *result)
{
	// Compute the global position of the thread in the grid
	int globalPos = threadIdx.x + blockIdx.x * blockDim.x;

	// Shared memory: One element per thread in the block
	// Call this kernel with the proper third launching parameter
	extern __shared__ int buffer[];

	// Load array values in the shared memory (0 if out of the array)
	if (globalPos < size)
	{
		if (array[globalPos].follows == globalPos)
		{
			buffer[threadIdx.x] = 1;
		}
		else
		{
			buffer[threadIdx.x] = 0;
		}
	}
	else
		buffer[threadIdx.x] = 0;

	// Wait for all the threads of the block to finish
	__syncthreads();

	// Reduction tree
	for (int step = blockDim.x / 2; step >= 1; step /= 2)
	{
		if (threadIdx.x < step)
			buffer[threadIdx.x] += buffer[threadIdx.x + step];
		__syncthreads();
	}

	// The maximum value of this block is on the first position of buffer
	if (threadIdx.x == 0)
		atomicAdd(result, buffer[0]);
}

__global__ void computeInitialitationMatrix(int *heights, int *trails, int size)
{

	// Compute the global position of the thread in the grid
	int globalPos = threadIdx.x + blockIdx.x * blockDim.x;

	if (globalPos < size)
	{
		heights[globalPos] = INT_MIN;
		trails[globalPos] = -1;
	}
}

#ifdef DEBUG
/*
 * Function: Print the current state of the simulation
 */
void print_heights(int rows, int columns, int *heights)
{
	/*
	 * You don't need to optimize this function, it is only for pretty
	 * printing and debugging purposes.
	 * It is not compiled in the production versions of the program.
	 * Thus, it is never used when measuring times in the leaderboard
	 */
	int i, j;
	printf("Heights:\n");
	printf("+");
	for (j = 0; j < columns; j++)
		printf("-------");
	printf("+\n");
	for (i = 0; i < rows; i++)
	{
		printf("|");
		for (j = 0; j < columns; j++)
		{
			char symbol;
			if (accessMat(heights, i, j) != INT_MIN)
				printf(" %6d", accessMat(heights, i, j));
			else
				printf("       ");
		}
		printf("|\n");
	}
	printf("+");
	for (j = 0; j < columns; j++)
		printf("-------");
	printf("+\n\n");
}

void print_trails(int rows, int columns, int *trails)
{
	/*
	 * You don't need to optimize this function, it is only for pretty
	 * printing and debugging purposes.
	 * It is not compiled in the production versions of the program.
	 * Thus, it is never used when measuring times in the leaderboard
	 */
	int i, j;
	printf("Trails:\n");
	printf("+");
	for (j = 0; j < columns; j++)
		printf("-------");
	printf("+\n");
	for (i = 0; i < rows; i++)
	{
		printf("|");
		for (j = 0; j < columns; j++)
		{
			char symbol;
			if (accessMat(trails, i, j) != -1)
				printf("%7d", accessMat(trails, i, j));
			else
				printf("       ", accessMat(trails, i, j));
		}
		printf("|\n");
	}
	printf("+");
	for (j = 0; j < columns; j++)
		printf("-------");
	printf("+\n\n");
}
#endif // DEBUG

/*
 * Function: Print usage line in stderr
 */
void show_usage(char *program_name)
{
	fprintf(stderr, "Usage: %s ", program_name);
	fprintf(stderr, "<rows> <columns> <x_min> <x_max> <y_min> <y_max> <searchers_density> <short_rnd1> <short_rnd2> <short_rnd3>\n");
	fprintf(stderr, "\n");
}

/*
 * MAIN PROGRAM
 */
int main(int argc, char *argv[])
{
	// This eliminates the buffer of stdout, forcing the messages to be printed immediately
	setbuf(stdout, NULL);

	int i;

	// Simulation data
	int rows, columns;	// Matrix sizes
	float x_min, x_max; // Limits of the terrain x coordinates
	float y_min, y_max; // Limits of the terrain y coordinates

	float searchers_density;	  // Density of hill climbing searchers
	unsigned short random_seq[3]; // Status of the random sequence

	int *heights; // Heights of the terrain points
	int *trails;  // Searchers trace and trails
	// int *tainted;		 // Position found in a search
	int num_searchers;	 // Number of searchers
	Searcher *searchers; // Searchers data
	int *total_steps;	 // Annotate accumulated steps to local maximums

	/* 1. Read simulation arguments */
	/* 1.1. Check minimum number of arguments */
	if (argc != 11)
	{
		fprintf(stderr, "-- Error: Not enough arguments when reading configuration from the command line\n\n");
		show_usage(argv[0]);
		exit(EXIT_FAILURE);
	}

	/* 1.2. Read argument values */
	rows = atoi(argv[1]);
	columns = atoi(argv[2]);
	x_min = atof(argv[3]);
	x_max = atof(argv[4]);
	y_min = atof(argv[5]);
	y_max = atof(argv[6]);
	searchers_density = atof(argv[7]);

	/* 1.3. Read random sequences initializer */
	for (i = 0; i < 3; i++)
	{
		random_seq[i] = (unsigned short)atoi(argv[8 + i]);
	}

#ifdef DEBUG
	/* 1.4. Print arguments */
	printf("Arguments, Rows: %d, Columns: %d\n", rows, columns);
	printf("Arguments, x_range: ( %d, %d ), y_range( %d, %d )\n", x_min, x_max, y_min, y_max);
	printf("Arguments, searchers_density: %f\n", searchers_density);
	printf("Arguments, Init Random Sequence: %hu,%hu,%hu\n", random_seq[0], random_seq[1], random_seq[2]);
	printf("\n");
#endif // DEBUG

	/* 2. Start global timer */
	CHECK_CUDA_CALL(cudaSetDevice(0));
	CHECK_CUDA_CALL(cudaDeviceSynchronize());
	double ttotal = cp_Wtime();

	/*
	 *
	 * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
	 *
	 */

	/* 3. Initialization */
	/* 3.1. Memory allocation */
	num_searchers = (int)(rows * columns * searchers_density);

	searchers = (Searcher *)malloc(sizeof(Searcher) * num_searchers);
	total_steps = (int *)malloc(sizeof(int) * num_searchers);
	if (searchers == NULL || total_steps == NULL)
	{
		fprintf(stderr, "-- Error allocating searchers structures for size: %d\n", num_searchers);
		exit(EXIT_FAILURE);
	}

	heights = (int *)malloc(sizeof(int) * (size_t)rows * (size_t)columns);
	trails = (int *)malloc(sizeof(int) * (size_t)rows * (size_t)columns);
	// tainted = (int *)malloc(sizeof(int) * (size_t)rows * (size_t)columns);
	if (heights == NULL || trails == NULL)
	{
		fprintf(stderr, "-- Error allocating terrain structures for size: %d x %d \n", rows, columns);
		exit(EXIT_FAILURE);
	}

	//**********************************************************//
	//					Pinned Memory			    			//
	//**********************************************************//
	/*
	cudaMallocHost( (void **)&searchers, sizeof(Searcher) * num_searchers); CHECK_CUDA_LAST();
	cudaMallocHost( (void **)&total_steps, sizeof(int) * num_searchers); CHECK_CUDA_LAST();

	cudaMallocHost( (void **)&heights, sizeof(int) * (size_t)rows * (size_t)columns); CHECK_CUDA_LAST();
	cudaMallocHost( (void **)&trails, sizeof(int) * (size_t)rows * (size_t)columns); CHECK_CUDA_LAST();
	cudaMallocHost( (void **)&tainted, sizeof(int) * (size_t)rows * (size_t)columns); CHECK_CUDA_LAST();
	*/

	//**********************************************************//
	//					Blocks & Threads		     			//
	//**********************************************************//
	// Declaración del numero de bloques y del numero de hilos por bloque
	int num_blocks;
	int num_blocks_reduction;
	int threads_per_block_reduction = 1024;
	int threads_per_block = 1024;

	// Calculamos el numero de bloques en funcion del tamaño del problema
	num_blocks = num_searchers / threads_per_block;
	num_blocks_reduction = (rows * columns) / threads_per_block_reduction;

	// Si el tamaño del vector no es multiplo del tamaño del bloque, lanzamos otro bloque completo (tendremos hilos sobrantes)
	if (num_searchers % threads_per_block != 0)
	{
		num_blocks++;
	}

	if ((rows * columns) % threads_per_block_reduction != 0)
	{
		num_blocks_reduction++;
	}

	//**********************************************************//
	//				Device			       //
	//**********************************************************//

	// Declaracion de los punteros donde vamos a almacenar la direccion de comienzo de nuestros datos en GPU
	int *dev_heights, *dev_trails;
	int *dev_total_steps;
	Searcher *dev_searchers;

	/* Reserva de memoria para los vectores en el DEVICE */
	// Empleamos cudaMalloc
	cudaMalloc((void **)&dev_total_steps, sizeof(int) * num_searchers);
	CHECK_CUDA_LAST();
	cudaMalloc((void **)&dev_searchers, sizeof(Searcher) * num_searchers);
	CHECK_CUDA_LAST();
	cudaMalloc((void **)&dev_heights, sizeof(int) * (size_t)rows * (size_t)columns);
	CHECK_CUDA_LAST();
	cudaMalloc((void **)&dev_trails, sizeof(int) * (size_t)rows * (size_t)columns);
	CHECK_CUDA_LAST();

	/* 3.1. Searchers initialization */
	int search;
	// Se mantiene por la aleatoriedad
	for (search = 0; search < num_searchers; search++)
	{
		searchers[search].pos_row = (int)(rows * erand48(random_seq));
		searchers[search].pos_col = (int)(columns * erand48(random_seq));
	}

	// Copia de datos del Host al Device
	//**********************************************************//
	//						HostToDevice			     		//
	//**********************************************************//

	/*
	cudaMemcpyAsync( dev_heights, heights, sizeof(int) * (size_t)rows * (size_t)columns, cudaMemcpyHostToDevice, stream[0]); CHECK_CUDA_LAST();
	cudaMemcpyAsync( dev_trails, trails, sizeof(int) * (size_t)rows * (size_t)columns, cudaMemcpyHostToDevice, stream[1]); CHECK_CUDA_LAST();
	cudaMemcpyAsync( dev_tainted, tainted, sizeof(int) * (size_t)rows * (size_t)columns, cudaMemcpyHostToDevice, stream[2]); CHECK_CUDA_LAST();
	cudaMemcpyAsync( dev_searchers, searchers, sizeof(Searcher) * num_searchers, cudaMemcpyHostToDevice, stream[3]); CHECK_CUDA_LAST();
	cudaMemcpyAsync( dev_total_steps, total_steps, sizeof(int) * num_searchers, cudaMemcpyHostToDevice, stream[4]); CHECK_CUDA_LAST();
	*/

	//cudaMemcpyAsync(dev_heights, heights, sizeof(int) * (size_t)rows * (size_t)columns, cudaMemcpyHostToDevice);
	//CHECK_CUDA_LAST();
	//cudaMemcpyAsync(dev_trails, trails, sizeof(int) * (size_t)rows * (size_t)columns, cudaMemcpyHostToDevice);
	//CHECK_CUDA_LAST();
	cudaMemcpyAsync(dev_searchers, searchers, sizeof(Searcher) * num_searchers, cudaMemcpyHostToDevice);
	CHECK_CUDA_LAST();
	//cudaMemcpyAsync(dev_total_steps, total_steps, sizeof(int) * num_searchers, cudaMemcpyHostToDevice);
	//CHECK_CUDA_LAST();

	//**********************************************************//
	//					Lanzamiento de Kernels	     			//
	//**********************************************************//

	// Creacion de Streams
	cudaStream_t stream[3];
	for (int i = 0; i < 3; i++)
	{
		cudaStreamCreate(&stream[i]);
	}
		

	/* 3.2. Terrain initialization */
	// Inicializamos las matrices del device
	computeInitialitationMatrix<<<num_blocks_reduction, threads_per_block_reduction, 0, stream[0]>>>(dev_heights, dev_trails, rows * columns);
	CHECK_CUDA_LAST();

	/* 3.3. Searchers initialization */
	searchersInitialization<<<num_blocks, threads_per_block, 0, stream[1]>>>(dev_searchers, dev_total_steps, num_searchers);
	CHECK_CUDA_LAST();

	/* 4. Compute searchers climbing trails */
	computeSearchersTrails<<<num_blocks, threads_per_block, 0, stream[2]>>>(num_searchers, rows, columns, dev_searchers, dev_heights, dev_trails, x_min, x_max, y_min, y_max);
	CHECK_CUDA_LAST();

	/*4.1*/
	computeFollows<<<num_blocks, threads_per_block, 0, stream[2]>>>(dev_searchers, dev_trails, num_searchers, columns);
	CHECK_CUDA_LAST();

	/* 5. Compute the leading follower of each searcher */
	computeFollower<<<num_blocks, threads_per_block, 0, stream[2]>>>(dev_searchers, num_searchers);
	CHECK_CUDA_LAST();

	/* 6. Compute accumulated trail steps to each maximum */
	computeAccumulatedTrail<<<num_blocks, threads_per_block, 0, stream[2]>>>(dev_searchers, dev_total_steps, num_searchers);
	CHECK_CUDA_LAST();

	/* 7. Compute statistical data */
	int num_local_max = 0;
	int max_height = INT_MIN;
	int max_accum_steps = INT_MIN;
	int total_tainted = 0;
	unsigned long long int total_heights = 0;

	int *max_steps;
	int *max_altura;
	int *max_local;
	unsigned long long int *alturas;
	int *tainted_prueba;

	// Shared Memory
	int shared_memory = threads_per_block * sizeof(int);
	int global_reduction = threads_per_block_reduction * sizeof(int);


	// Maximo accum steps
	cudaMemsetAsync(max_steps, INT_MIN, sizeof(int));
	cudaMalloc(&max_steps, sizeof(int)); CHECK_CUDA_LAST();
	reductionMax<<<num_blocks, threads_per_block, shared_memory>>>(dev_total_steps, num_searchers, max_steps); CHECK_CUDA_LAST();
	cudaMemcpyAsync(&max_accum_steps, max_steps, sizeof(int), cudaMemcpyDeviceToHost); CHECK_CUDA_LAST();

	// Maxima altura ++ Total Heights
	cudaMalloc(&max_altura, sizeof(int)); CHECK_CUDA_LAST();
	cudaMalloc(&alturas, sizeof(unsigned long long int)); CHECK_CUDA_LAST();
	reductionMax<<<num_blocks_reduction, threads_per_block_reduction, global_reduction>>>(dev_heights, rows * columns, max_altura); CHECK_CUDA_LAST();
	reductionAddHeights<<<num_blocks_reduction, threads_per_block_reduction, global_reduction>>>(dev_heights, rows * columns, alturas); CHECK_CUDA_LAST();
	cudaMemcpyAsync(&max_height, max_altura, sizeof(int), cudaMemcpyDeviceToHost); CHECK_CUDA_LAST();
	cudaMemcpyAsync(&total_heights, alturas, sizeof(unsigned long long int), cudaMemcpyDeviceToHost); CHECK_CUDA_LAST();

	// Num local max	
	cudaMalloc(&max_local, sizeof(int)); CHECK_CUDA_LAST();
	computeLocalMax<<<num_blocks, threads_per_block, shared_memory>>>(dev_searchers, num_searchers, max_local); CHECK_CUDA_LAST();
	cudaMemcpyAsync(&num_local_max, max_local, sizeof(int), cudaMemcpyDeviceToHost); CHECK_CUDA_LAST();

	// Total tainted
	cudaMalloc(&tainted_prueba, sizeof(int)); CHECK_CUDA_LAST();
	reductionAdd<<<num_blocks_reduction, threads_per_block_reduction, global_reduction>>>(dev_trails, rows * columns, tainted_prueba); CHECK_CUDA_LAST();
	cudaMemcpyAsync(&total_tainted, tainted_prueba, sizeof(int), cudaMemcpyDeviceToHost); CHECK_CUDA_LAST();

	// Sincronizacion final de hilos
	cudaDeviceSynchronize();

	// Destruccion de streams
	for (int i = 0; i < 3; i++)
	{
		cudaStreamDestroy(stream[i]);
	}

#ifdef DEBUG
#ifdef _OPENMP
	/* Print computed heights at the end of the search */
	print_heights(rows, columns, heights);
#endif
#endif

	// Liberacion de Memoria
	cudaFree(dev_total_steps);
	CHECK_CUDA_LAST();
	cudaFree(dev_searchers);
	CHECK_CUDA_LAST();
	cudaFree(dev_heights);
	CHECK_CUDA_LAST();
	cudaFree(dev_trails);
	CHECK_CUDA_LAST();

	cudaFree(max_steps);
	CHECK_CUDA_LAST();
	cudaFree(tainted_prueba);
	CHECK_CUDA_LAST();
	cudaFree(max_altura);
	CHECK_CUDA_LAST();
	cudaFree(max_local);
	CHECK_CUDA_LAST();
	cudaFree(alturas);
	CHECK_CUDA_LAST();

	/*
	 *
	 * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
	 *
	 */

	/* 5. Stop global time */
	CHECK_CUDA_CALL(cudaDeviceSynchronize());
	ttotal = cp_Wtime() - ttotal;

	/* 6. Output for leaderboard */
	printf("\n");
	/* 6.1. Total computation time */
	printf("Time: %lf\n", ttotal);

	/* 6.2. Results: Statistics */
	printf("Result: %d, %d, %d, %d, %llu\n\n",
		   num_local_max,
		   max_height,
		   max_accum_steps,
		   total_tainted,
		   total_heights);

	/* 7. Free resources */

	free(searchers);
	free(total_steps);
	free(heights);
	free(trails);
	// free(tainted);
	/*
	cudaFreeHost( searchers );
	cudaFreeHost( total_steps );
	cudaFreeHost( heights );
	cudaFreeHost( trails );
	cudaFreeHost( tainted );
	*/

	/* 8. End */
	return 0;
}
