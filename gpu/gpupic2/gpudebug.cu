#include <stdlib.h>
#include <stdio.h>
#include "cuda.h"

extern int nblock_size;
extern int maxgsx;

static cudaError_t crc;


__global__ void printtile(float ppart[], float ppbuff[], int kpic[],
                           int ncl[], int ihole[], int idimp, int nppmx,
                           int mx1, int my1, int npbmx, int ntmax,
                           int *irc, int idx, unsigned long long int *psum, int details) {
/* this subroutine performs third step of a particle sort by x,y grid
   in tiles of mx, my, where incoming particles from other tiles are
   copied into ppart.
   linear interpolation, with periodic boundary conditions
   tiles are assumed to be arranged in 2D linear memory
   input: all except irc
   output: ppart, kpic, irc
   ppart[k][i][n] = i co-ordinate of particle n in tile k
   ppbuff[k][i][n] = i co-ordinate of particle n in tile k
   kpic[k] = number of particles in tile k
   ncl[k] = number of particles departing from tile k
   ihole[0][k][:] = location of hole in array left by departing particle
   ihole[1][k][:] = buffer for real holes at the end of ihole[0][k][:]
   ihole[0][k][0] = ih, number of holes left (error, if negative)
   idimp = size of phase space = 4
   nppmx = maximum number of particles in tile
   mx1 = (system length in x direction - 1)/mx + 1
   my1 = (system length in y direction - 1)/my + 1
   npbmx = size of buffer array ppbuff
   ntmax = size of hole array for particles leaving tiles
   irc = maximum overflow, returned only if error occurs, when irc > 0
local data                                                            */
   int mxy1, k, i, np, st, ed;
   unsigned long long int checksum;
/* The sizes of the shared memory arrays are as follows: */
   mxy1 = mx1*my1;
/* k = tile number */
   k = blockIdx.x + gridDim.x*blockIdx.y;
   checksum = 0;
if (k==idx){
		//printf("ihole=\n");
	if (threadIdx.x == 0){
	      printf("\nk=%d,nhp=%d,ncl=%d,rh=%d,kpic=%d,irc=%d\n",k,ihole[(ntmax+1)*k], ncl[k], ihole[(ntmax+1)*(k+mxy1)], kpic[k], *irc);
	   if (details > 0){
	      printf("\n--iholes k=%10d--------------------------\n", k);
	      for (i = 0; i<ihole[(ntmax+1)*k]; i++)
	        printf("%10d\t", ihole[(ntmax+1)*k+i+1]);
	      printf("\n--ihole buffer k=%10d-----------------------\n", k);
	      for (i = 0; i<ihole[(ntmax+1)*k]; i++)
	        printf("%10d\t", ihole[(ntmax+1)*(k+mxy1)+i]);
	      printf("\n--particles@iholes pos. k=%10d------------------\n", k);
	      for (i = 0; i<ihole[(ntmax+1)*k]; i++)
	        printf("%10d\t", int(ppart[ihole[(ntmax+1)*k+i+1]-1+nppmx*(4+idimp*k)]));
	      np = ncl[k] - ihole[(ntmax+1)*k];
	      printf("\n--particles@the end %d pos. k=%10d------------------\n", np, k);
	      if (np < 0) {
		      st = kpic[k];
		      ed = kpic[k] - np;
	      }
	      else {
		      st = kpic[k] - np;
		      ed = kpic[k];
	      }
	      for (i = st; i<ed; i++)
	        printf("%10d\t", int(ppart[i+nppmx*(4+idimp*k)]));
              printf("\n--incoming particles k=%10d-----------------------\n", k);
	      for (i = 0; i<ncl[k]; i++)
	        printf("%10d\t", int(ppbuff[i+npbmx*(4+idimp*k)]));
	      printf("\n\n\n");
	   }
      for (i = 0; i<kpic[k]; i++){
	 //printf("%10d\t", int(ppart[i+nppmx*(4+idimp*k)]));
         checksum += int(ppart[i+nppmx*(4+idimp*k)]);
      }
	atomicAdd(psum,checksum);
	}
	__threadfence_block();
	__syncthreads();
	
}

}

__global__ void checkduplicates(float ppart[], int pidxbuff[], int kpic[],
                           int ncl[], int ihole[], int idimp, int nppmx,
                           int mx1, int my1, int npbmx, int ntmax,
                           int *irc, int idx, int check) {
	int k, i, pidx;
   k = blockIdx.x + gridDim.x*blockIdx.y;
if (k==idx && threadIdx.x == 0 && check != 0){
      for (i = 0; i<kpic[k]; i++){
	 //printf("%10d\t", int(ppart[i+nppmx*(4+idimp*k)]));
         pidx = int(ppart[i+nppmx*(4+idimp*k)]);
	 if (pidxbuff[pidx] == 0)
            pidxbuff[pidx] = pidx;
	 else {
	    printf("ERROR FOUND! k=%d, particle idx=%d, kpic=%d, particle position=%d, pidxbuff=%d\n", k, pidx, kpic[k], i+1, pidxbuff[pidx]);
	    *irc = 1;
	 }
      }
}
}

__global__ void resetcounters(int *ncl, int *irc, int mxy1) {
	int k, i;
   k = blockIdx.x + gridDim.x*blockIdx.y;
if (k==0 && threadIdx.x == 0) {
  for (i=0; i<mxy1; i++)
	   ncl[i] = 0;
}
}

__global__ void resetirc(int *irc) {
	int k;
   k = blockIdx.x + gridDim.x*blockIdx.y;
if (k==0 && threadIdx.x == 0) {
	*irc = 0;
}
}

__global__ void checksum(unsigned long long int *psum) {
	int k;
   k = blockIdx.x + gridDim.x*blockIdx.y;
if (k==0 && threadIdx.x == 0)
   printf("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx %10ld >>>>>>>>>>>>\n",*psum);
}

__global__ void resetpsum(unsigned long long int *psum, int mxy1) {
//   if (*irc == 8390656)
      *psum = 0;
}

__global__ void resetpid(float ppart[], float ppbuff[], int kpic[],
                           int ncl[], int ihole[], int idimp, int nppmx,
                           int mx1, int my1, int npbmx, int ntmax,
                           int *irc) {
   int mxy1, k, i, j, checksum;
/* The sizes of the shared memory arrays are as follows: */
   mxy1 = mx1*my1;
/* k = tile number */
   k = blockIdx.x + gridDim.x*blockIdx.y;
   if (threadIdx.x == 0) {
      i = kpic[k];
      j = 1;
      while (i>0) {
         i = i/10;
         j *= 10;
      }
      for (i = 0; i<kpic[k]; i++){
         ppart[i+nppmx*(4+idimp*k)] = j*k + i + 1;
      }
   }
   
}

/*--------------------------------------------------------------------*/
extern "C" void resetparticleid(float *ppart, float *ppbuff, int *kpic,
                            int *ncl, int *ihole, int idimp, int nppmx,
                            int nx, int ny, int mx, int my, int mx1,
                            int my1, int npbmx, int ntmax, int *irc) {
int mxy1, m, n, ns, i;
   dim3 dimBlock(nblock_size);
   mxy1 = mx1*my1;
   m = (mxy1 - 1)/maxgsx + 1;
   n = mxy1 < maxgsx ? mxy1 : maxgsx;
   dim3 dimGrid(n,m);
//   printf("mxy1=%d\n",mxy1);
/* buffer particles that are leaving tile and sum ncl */
   ns = 9*sizeof(int);
//   crc = cudaGetLastError();
//   gpuppmov2l<<<dimGrid,dimBlock,ns>>>(ppart,ppbuff,ncl,ihole,idimp,
//                                       nppmx,mx1,my1,npbmx,ntmax,irc);
/* cudaDeviceSynchronize(); */
 //  crc = cudaGetLastError();
//   if (crc) {
//      printf("gpuppmov2l error=%d:%s\n",crc,cudaGetErrorString(crc));
//      exit(1);
//   }
/* copy incoming particles from ppbuff into ppart, update kpic */
   ns = (nblock_size+18)*sizeof(int);
   cudaDeviceSynchronize();
   resetpid<<<dimGrid,dimBlock,ns>>>(ppart,ppbuff,kpic,ncl,ihole,
                                       idimp,nppmx,mx1,my1,npbmx,ntmax,
                                       irc);
}

void gpu_lallocate(unsigned long long int **g_f, int nsize, int *irc) {
/* allocate global float memory on GPU, return pointer to C */
   void *gptr;
   crc = cudaMalloc(&gptr,sizeof(unsigned long long int)*nsize);
   if (crc) {
      printf("cudaMalloc float Error=%d:%s,l=%d\n",crc,
              cudaGetErrorString(crc),nsize);
      *irc = 1;
   }
   *g_f = (unsigned long long int *)gptr;
   //crc = cudaMemset((void *)g_f,0,sizeof(unsigned long long int)*nsize);
   //if (crc) {
   //   printf("cudaMemset Error=%d:%s\n",crc,cudaGetErrorString(crc));
   //   exit(1);
   //}
   return;
}

void gpu_deallocate(void *g_d, int *irc) {
/* deallocate global memory on GPU */
   crc = cudaFree(g_d);
   if (crc) {
      printf("cudaFree Error=%d:%s\n",crc,cudaGetErrorString(crc));
      *irc = 1;
   }
   return;
}

void gpu_iallocate(int **g_f, long nsize, int *irc) {
/* allocate global float memory on GPU, return pointer to C */
   void *gptr;
   crc = cudaMalloc(&gptr,sizeof(int)*nsize);
   if (crc) {
      printf("cudaMalloc float Error=%d:%s,l=%d\n",crc,
              cudaGetErrorString(crc),nsize);
      *irc = 1;
   }
   crc = cudaMemset(gptr,0,sizeof(int)*nsize);
   if (crc) {
      printf("cudaMemset Error=%d:%s\n",crc,cudaGetErrorString(crc));
      exit(1);
   }
   *g_f = (int *)gptr;
   return;
}

/*--------------------------------------------------------------------*/
void gpu_icopyout(int *f, int *g_f, int nsize) {
/* copy int array from global GPU memory to host memory */
   crc = cudaMemcpy(f,(void *)g_f,sizeof(int)*nsize,
                    cudaMemcpyDeviceToHost);
   if (crc) {
      printf("cudaMemcpyDeviceToHost int Error=%d:%s\n",crc,
              cudaGetErrorString(crc));
      exit(1);
   }
   return;
}

/*--------------------------------------------------------------------*/
extern "C" void cgpuprinttile(float *ppart, float *ppbuff, int *kpic,
                            int *ncl, int *ihole, int idimp, int nppmx,
                            int nx, int ny, int mx, int my, int mx1,
                            int my1, int npbmx, int ntmax, int *irc, int stage) {
int mxy1, m, n, ns, i;
int err;
unsigned long long int *psum = NULL;
int *pidxbuff = NULL, *npar = NULL;
long totpar = 0;
   dim3 dimBlock(nblock_size);
   mxy1 = mx1*my1;
   m = (mxy1 - 1)/maxgsx + 1;
   n = mxy1 < maxgsx ? mxy1 : maxgsx;
   dim3 dimGrid(n,m);
//   printf("mxy1=%d\n",mxy1);
/* buffer particles that are leaving tile and sum ncl */
   ns = 9*sizeof(int);
//   crc = cudaGetLastError();
//   gpuppmov2l<<<dimGrid,dimBlock,ns>>>(ppart,ppbuff,ncl,ihole,idimp,
//                                       nppmx,mx1,my1,npbmx,ntmax,irc);
/* cudaDeviceSynchronize(); */
 //  crc = cudaGetLastError();
//   if (crc) {
//      printf("gpuppmov2l error=%d:%s\n",crc,cudaGetErrorString(crc));
//      exit(1);
//   }
/* copy incoming particles from ppbuff into ppart, update kpic */
   gpu_lallocate(&psum,1,&err);
   ns = (nblock_size+18)*sizeof(int);

   /* check for duplicates partiles in tiles >>> */
   npar = (int *) calloc(mxy1, sizeof(int));
   gpu_icopyout(npar, kpic, mxy1);
   for (i=0; i<mxy1; i++)
	   totpar += npar[i];
   resetirc<<<dimGrid,dimBlock,ns>>>(irc);
   gpu_iallocate(&pidxbuff, totpar+1, &err);
   for (i=0; i<mxy1; i++){
   cudaDeviceSynchronize();
   checkduplicates<<<dimGrid,dimBlock,ns>>>(ppart,pidxbuff,kpic,ncl,ihole,
                                       idimp,nppmx,mx1,my1,npbmx,ntmax,
                                       irc,i,stage);
   }
   gpu_icopyout(&err, irc, 1);
   /* <<< check for duplicates partiles in tiles */

   cudaDeviceSynchronize();
   resetpsum<<<dimGrid,dimBlock,ns>>>(psum, mxy1);
  // crc = cudaGetLastError();
   for (i=0; i<mxy1; i++){
   cudaDeviceSynchronize();
   printtile<<<dimGrid,dimBlock,ns>>>(ppart,ppbuff,kpic,ncl,ihole,
                                       idimp,nppmx,mx1,my1,npbmx,ntmax,
                                       irc,i,psum,0);
   }

   cudaDeviceSynchronize();
   checksum<<<dimGrid,dimBlock,ns>>>(psum);
   cudaDeviceSynchronize();
   if (stage == 0) mxy1 = 0;
   resetcounters<<<dimGrid,dimBlock,ns>>>(ncl, irc, mxy1);
   
   cudaDeviceSynchronize();
   gpu_deallocate((void *)psum,&err);
   gpu_deallocate((void *)pidxbuff,&err);
}
