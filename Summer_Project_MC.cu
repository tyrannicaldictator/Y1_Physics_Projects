//Author: Minseo Kim 
//This code contains the data structures and functions for Monte-Carlo simulations
//to find the final positions of Uranus and Neptune after 150 years given an initial perturbation 
//accounting for measurement uncertainties in initial positions of the planets,
//as well as the functions for parallel computing. 

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <curand_kernel.h>
#include "Header.h"

#define N_BODIES 9

cudaError_t limitErr = cudaDeviceSetLimit(cudaLimitStackSize, 8192);
__constant__ double G = 6.67430e-11;

// =====================================================================
// GPU DATA STRUCTURES
// =====================================================================
struct DeviceSystemState {
    double x[N_BODIES], y[N_BODIES], z[N_BODIES];
    double vx[N_BODIES], vy[N_BODIES], vz[N_BODIES];
};

struct DeviceVec3 {
    double x, y, z;
    __device__ DeviceVec3(double x, double y, double z) : x(x), y(y), z(z) {}
    
    __device__ double mag() const { return sqrt(x*x + y*y + z*z); }
    
    __device__ DeviceVec3 normalize() const {
        double m = mag();
        return DeviceVec3(x/m, y/m, z/m);
    }
    
    __device__ DeviceVec3 cross(const DeviceVec3& other) const {
        return DeviceVec3(y * other.z - z * other.y,
                          z * other.x - x * other.z,
                          x * other.y - y * other.x);
    }
};

// =====================================================================
// GPU PHYSICS ENGINE (__device__ functions run on the GPU)
// =====================================================================
__device__ void get_derivative(const DeviceSystemState& s, DeviceSystemState& deriv, const double* masses) {
    for (int i = 0; i < N_BODIES; ++i) {
        deriv.x[i] = s.vx[i];
        deriv.y[i] = s.vy[i];
        deriv.z[i] = s.vz[i];
        
        double ax = 0, ay = 0, az = 0;
        for (int j = 0; j < N_BODIES; ++j) {
            if (i == j) continue;

            double dx = s.x[j] - s.x[i];
            double dy = s.y[j] - s.y[i];
            double dz = s.z[j] - s.z[i];
            
            double r2 = dx*dx + dy*dy + dz*dz;
            double r = sqrt(r2);
            double r3 = r2 * r;

            double a_mag = (G * masses[j]) / r3;
            
            ax += a_mag * dx;
            ay += a_mag * dy;
            az += a_mag * dz;
        }
        deriv.vx[i] = ax;
        deriv.vy[i] = ay;
        deriv.vz[i] = az;
    }
}

__device__ void rk4_step(DeviceSystemState& s, const double* masses, double dt) {
    DeviceSystemState k1, k2, k3, k4, temp;

    get_derivative(s, k1, masses);

    for(int i=0; i<N_BODIES; ++i) {
        temp.x[i] = s.x[i] + k1.x[i] * (dt / 2.0); temp.vx[i] = s.vx[i] + k1.vx[i] * (dt / 2.0);
        temp.y[i] = s.y[i] + k1.y[i] * (dt / 2.0); temp.vy[i] = s.vy[i] + k1.vy[i] * (dt / 2.0);
        temp.z[i] = s.z[i] + k1.z[i] * (dt / 2.0); temp.vz[i] = s.vz[i] + k1.vz[i] * (dt / 2.0);
    }
    get_derivative(temp, k2, masses);

    for(int i=0; i<N_BODIES; ++i) {
        temp.x[i] = s.x[i] + k2.x[i] * (dt / 2.0); temp.vx[i] = s.vx[i] + k2.vx[i] * (dt / 2.0);
        temp.y[i] = s.y[i] + k2.y[i] * (dt / 2.0); temp.vy[i] = s.vy[i] + k2.vy[i] * (dt / 2.0);
        temp.z[i] = s.z[i] + k2.z[i] * (dt / 2.0); temp.vz[i] = s.vz[i] + k2.vz[i] * (dt / 2.0);
    }
    get_derivative(temp, k3, masses);

    for(int i=0; i<N_BODIES; ++i) {
        temp.x[i] = s.x[i] + k3.x[i] * dt; temp.vx[i] = s.vx[i] + k3.vx[i] * dt;
        temp.y[i] = s.y[i] + k3.y[i] * dt; temp.vy[i] = s.vy[i] + k3.vy[i] * dt;
        temp.z[i] = s.z[i] + k3.z[i] * dt; temp.vz[i] = s.vz[i] + k3.vz[i] * dt;
    }
    get_derivative(temp, k4, masses);

    for(int i=0; i<N_BODIES; ++i) {
        s.x[i] += (k1.x[i] + k2.x[i]*2.0 + k3.x[i]*2.0 + k4.x[i]) * (dt / 6.0);
        s.y[i] += (k1.y[i] + k2.y[i]*2.0 + k3.y[i]*2.0 + k4.y[i]) * (dt / 6.0);
        s.z[i] += (k1.z[i] + k2.z[i]*2.0 + k3.z[i]*2.0 + k4.z[i]) * (dt / 6.0);
        s.vx[i] += (k1.vx[i] + k2.vx[i]*2.0 + k3.vx[i]*2.0 + k4.vx[i]) * (dt / 6.0);
        s.vy[i] += (k1.vy[i] + k2.vy[i]*2.0 + k3.vy[i]*2.0 + k4.vy[i]) * (dt / 6.0);
        s.vz[i] += (k1.vz[i] + k2.vz[i]*2.0 + k3.vz[i]*2.0 + k4.vz[i]) * (dt / 6.0);
    }
}

__device__ void shift_to_barycenter(DeviceSystemState& s, const double* masses) {
    double total_mass = 0, cx = 0, cy = 0, cz = 0, cvx = 0, cvy = 0, cvz = 0;
    for (int i = 0; i < N_BODIES; ++i) {
        total_mass += masses[i];
        cx += masses[i] * s.x[i];       cy += masses[i] * s.y[i];       cz += masses[i] * s.z[i];
        cvx += masses[i] * s.vx[i];     cvy += masses[i] * s.vy[i];     cvz += masses[i] * s.vz[i];
    }
    cx /= total_mass; cy /= total_mass; cz /= total_mass;
    cvx /= total_mass; cvy /= total_mass; cvz /= total_mass;
    for (int i = 0; i < N_BODIES; ++i) {
        s.x[i] -= cx;   s.y[i] -= cy;   s.z[i] -= cz;
        s.vx[i] -= cvx; s.vy[i] -= cvy; s.vz[i] -= cvz;
    }
}

// =====================================================================
// THE CUDA KERNEL (Runs on GPU, launched from CPU)
// =====================================================================
__global__ void __launch_bounds__(32) nBodyMonteCarloKernel(const DeviceSystemState* initial_state, 
                                      const double* masses, 
                                      double* d_results, 
                                      int total_steps, double dt, int num_mc_runs) {
    
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    if (threadId >= num_mc_runs) return;

    // Load initial state into local thread registers
    DeviceSystemState s = *initial_state;

    // Initialize parallel RNG
    curandState state;
    curand_init(1337, threadId, 0, &state);

    // Uranus (Index 7) and Neptune (Index 8) Noise Params
    double ura_sigma_R = 100.0 * 1000.0, ura_sigma_T = 3000.0 * 1000.0, ura_sigma_N = 100.0 * 1000.0;
    double nep_sigma_R = 100.0 * 1000.0, nep_sigma_T = 3000.0 * 1000.0, nep_sigma_N = 100.0 * 1000.0;

    // Construct RTN frames
    DeviceVec3 r_ura(s.x[7], s.y[7], s.z[7]), v_ura(s.vx[7], s.vy[7], s.vz[7]);
    DeviceVec3 r_nep(s.x[8], s.y[8], s.z[8]), v_nep(s.vx[8], s.vy[8], s.vz[8]);

    DeviceVec3 R_ura = r_ura.normalize(), N_ura = r_ura.cross(v_ura).normalize(), T_ura = N_ura.cross(R_ura);
    DeviceVec3 R_nep = r_nep.normalize(), N_nep = r_nep.cross(v_nep).normalize(), T_nep = N_nep.cross(R_nep);

    // Apply anisotropic perturbations using standard normal * sigma
    s.x[7] += (curand_normal_double(&state)*ura_sigma_R*R_ura.x) + (curand_normal_double(&state)*ura_sigma_T*T_ura.x) + (curand_normal_double(&state)*ura_sigma_N*N_ura.x);
    s.y[7] += (curand_normal_double(&state)*ura_sigma_R*R_ura.y) + (curand_normal_double(&state)*ura_sigma_T*T_ura.y) + (curand_normal_double(&state)*ura_sigma_N*N_ura.y);
    s.z[7] += (curand_normal_double(&state)*ura_sigma_R*R_ura.z) + (curand_normal_double(&state)*ura_sigma_T*T_ura.z) + (curand_normal_double(&state)*ura_sigma_N*N_ura.z);
    
    s.x[8] += (curand_normal_double(&state)*nep_sigma_R*R_nep.x) + (curand_normal_double(&state)*nep_sigma_T*T_nep.x) + (curand_normal_double(&state)*nep_sigma_N*N_nep.x);
    s.y[8] += (curand_normal_double(&state)*nep_sigma_R*R_nep.y) + (curand_normal_double(&state)*nep_sigma_T*T_nep.y) + (curand_normal_double(&state)*nep_sigma_N*N_nep.y);
    s.z[8] += (curand_normal_double(&state)*nep_sigma_R*R_nep.z) + (curand_normal_double(&state)*nep_sigma_T*T_nep.z) + (curand_normal_double(&state)*nep_sigma_N*N_nep.z);

    shift_to_barycenter(s, masses);

    // Massive parallel RK4 Integration
    for (int step = 0; step < total_steps; ++step) {
        rk4_step(s, masses, dt);
    }

    // Write final coordinates of Uranus and Neptune to global memory output array
    // Layout: [Ura_X, Ura_Y, Ura_Z, Nep_X, Nep_Y, Nep_Z] contiguous per thread
    int idx = threadId * 6;
    d_results[idx] = s.x[7];     d_results[idx+1] = s.y[7];   d_results[idx+2] = s.z[7];
    d_results[idx+3] = s.x[8];   d_results[idx+4] = s.y[8];   d_results[idx+5] = s.z[8];
}

// =====================================================================
// CPU WRAPPER (Handles GPU memory allocation and execution)
// =====================================================================
void runMonteCarloSimulation() {

    cudaDeviceSetLimit(cudaLimitStackSize, 8192);

    int num_mc_runs = 500;
    double years = 150;
    double dt = 3600*8;
    int total_steps = (3600 * 24 * 365 * years) / dt;

    // 1. Setup Host (CPU) Initial State
    double h_masses[N_BODIES] = { 1.989e30, 3.301e23, 4.867e24, 5.972e24, 6.417e23, 1.898e27, 5.683e26, 8.681e25, 1.024e26 };
    DeviceSystemState h_state;
    double init_x[] = { 4.896e+05, -4.896e+07, -9.806e+07, -1.097e+08, 1.077e+08, -6.495e+08, -7.262e+08, 2.867e+09, 4.140e+09 };
    double init_y[] = { -2.380e+04, 1.982e+07, 4.256e+07, -1.029e+08, 1.968e+08, 4.687e+08, -1.302e+09, 8.516e+08, -1.718e+09 };
    double init_z[] = { -2.216e+04, 6.136e+06, 6.249e+06, -1.863e+04, 1.470e+06, 1.257e+07, 5.154e+07, -3.399e+07, -6.003e+07 };
    double init_vx[] = { 5.154e-03, -28.18, -14.04, 19.83, -20.34, -7.80, 7.90, -1.98, 2.04 };
    double init_vy[] = { 1.049e-02, -43.12, -32.30, -21.87, 13.66, -9.97, -4.73, 6.21, 5.05 };
    double init_vz[] = { -1.351e-04, -0.938, 0.367, -0.00038, 0.785, 0.216, -0.232, 0.048, -0.150 };

    for(int i=0; i<N_BODIES; i++) {
        h_state.x[i] = init_x[i]*1000; h_state.y[i] = init_y[i]*1000; h_state.z[i] = init_z[i]*1000;
        h_state.vx[i] = init_vx[i]*1000; h_state.vy[i] = init_vy[i]*1000; h_state.vz[i] = init_vz[i]*1000;
    }

    // 2. Allocate Device (GPU) Memory
    DeviceSystemState* d_state;
    double *d_masses, *d_results;
    cudaMalloc(&d_state, sizeof(DeviceSystemState));
    cudaMalloc(&d_masses, N_BODIES * sizeof(double));
    cudaMalloc(&d_results, num_mc_runs * 6 * sizeof(double));

    cudaMemcpy(d_state, &h_state, sizeof(DeviceSystemState), cudaMemcpyHostToDevice);
    cudaMemcpy(d_masses, h_masses, N_BODIES * sizeof(double), cudaMemcpyHostToDevice);

    // 3. Execution Configuration
    int threadsPerBlock = 32;
    int blocksPerGrid = (num_mc_runs + threadsPerBlock - 1) / threadsPerBlock;

// 3. Launch the Kernel
    std::cout << "Pushing " << num_mc_runs << " trials to GPU. Stand by..." << std::endl;
    nBodyMonteCarloKernel<<<blocksPerGrid, threadsPerBlock>>>(d_state, d_masses, d_results, total_steps, dt, num_mc_runs);
    
    // Catch launch errors (e.g., bad grid dimensions)
    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        // Cast launchErr to an (int) to get the raw numeric code
        std::cout << "FATAL LAUNCH ERROR CODE [" << (int)launchErr << "]: " << cudaGetErrorString(launchErr) << std::endl;
    }

    // Catch execution errors (e.g., memory violations, stack overflows)
    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        std::cout << "FATAL EXECUTION ERROR: " << cudaGetErrorString(syncErr) << std::endl;
    }
    nBodyMonteCarloKernel<<<blocksPerGrid, threadsPerBlock>>>(d_state, d_masses, d_results, total_steps, dt, num_mc_runs);
    cudaDeviceSynchronize();

    // 4. Retrieve Results and Write to CSV
    double* h_results = new double[num_mc_runs * 6];
    cudaMemcpy(h_results, d_results, num_mc_runs * 6 * sizeof(double), cudaMemcpyDeviceToHost);

    std::ofstream outFile("uranus_neptune_mc_endpoints.csv");
    outFile << "run,final_x_ura,final_y_ura,final_z_ura,final_x_nep,final_y_nep,final_z_nep\n";
    for(int i = 0; i < num_mc_runs; i++) {
        int idx = i * 6;
        outFile << i << "," << h_results[idx] << "," << h_results[idx+1] << "," << h_results[idx+2] 
                << "," << h_results[idx+3] << "," << h_results[idx+4] << "," << h_results[idx+5] << "\n";
    }
    outFile.close();

    // 5. Cleanup
    cudaFree(d_state); cudaFree(d_masses); cudaFree(d_results); delete[] h_results;
}
